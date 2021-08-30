from typing import Tuple
from tqdm import tqdm

import torch
from torch import tensor
from torch.utils.data import DataLoader
import timm

import numpy as np
from sklearn.metrics import roc_auc_score

from utils import GaussianBlur, get_coreset_idx_randomp, get_tqdm_params


class KNNExtractor(torch.nn.Module):
	def __init__(
		self,
		backbone_name : str = "resnet50",
		out_indices : Tuple = None,
		pool_last : bool = False,
	):
		super().__init__()

		self.feature_extractor = timm.create_model(
			backbone_name,
			out_indices=out_indices,
			features_only=True,
			pretrained=True,
		)
		for param in self.feature_extractor.parameters():
			param.requires_grad = False
		self.feature_extractor.eval()
		
		self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool_last else None
		self.backbone_name = backbone_name # for results metadata
		self.out_indices = out_indices

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.feature_extractor = self.feature_extractor.to(self.device)
			
	def __call__(self, x: tensor):
		with torch.no_grad():
			feature_maps = self.feature_extractor(x.to(self.device))
		feature_maps = [fmap.to("cpu") for fmap in feature_maps]
		if self.pool:
			# spit into fmaps and z
			return feature_maps[:-1], self.pool(feature_maps[-1])
		else:
			return feature_maps

	def fit(self, _: DataLoader):
		raise NotImplementedError

	def predict(self, _: tensor):
		raise NotImplementedError

	def evaluate(self, test_dl: DataLoader) -> Tuple[float, float]:
		"""Calls predict step for each test sample."""
		image_preds = []
		image_labels = []
		pixel_preds = []
		pixel_labels = []

		for sample, mask, label in tqdm(test_dl, **get_tqdm_params()):
			z_score, fmap = self.predict(sample)
			
			image_preds.append(z_score.numpy())
			image_labels.append(label)
			
			pixel_preds.extend(fmap.flatten().numpy())
			pixel_labels.extend(mask.flatten().numpy())
			
		image_preds = np.stack(image_preds)

		image_rocauc = roc_auc_score(image_labels, image_preds)
		pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)

		return image_rocauc, pixel_rocauc

	def get_parameters(self, extra_params : dict = None) -> dict:
		return {
			"backbone_name": self.backbone_name,
			"out_indices": self.out_indices,
			**extra_params,
		}

class SPADE(KNNExtractor):
	def __init__(
		self,
		k: int = 5,
		backbone_name: str = "resnet18",
	):
		super().__init__(
			backbone_name=backbone_name,
			out_indices=(1,2,3,-1),
			pool_last=True,
		)
		self.k = k
		self.image_size = 224
		self.z_lib = []
		self.feature_maps = []
		self.threshold_z = None
		self.threshold_fmaps = None
		self.blur = GaussianBlur(4)

	def fit(self, train_dl):
		for sample, _ in tqdm(train_dl, **get_tqdm_params()):
			feature_maps, z = self(sample)

			# z vector
			self.z_lib.append(z)

			# feature maps
			if len(self.feature_maps) == 0:
				for fmap in feature_maps:
					self.feature_maps.append([fmap])
			else:
				for idx, fmap in enumerate(feature_maps):
					self.feature_maps[idx].append(fmap)

		self.z_lib = torch.vstack(self.z_lib)
		
		for idx, fmap in enumerate(self.feature_maps):
			self.feature_maps[idx] = torch.vstack(fmap)

	def predict(self, sample):
		feature_maps, z = self(sample)

		distances = torch.linalg.norm(self.z_lib - z, dim=1)
		values, indices = torch.topk(distances.squeeze(), self.k, largest=False)

		z_score = values.mean()

		# Build the feature gallery out of the k nearest neighbours.
		# The authors migh have concatenated all features maps first, then check the minimum norm per pixel.
		# Here, we check for the minimum norm first, then concatenate (sum) in the final layer.
		scaled_s_map = torch.zeros(1,1,self.image_size,self.image_size)
		for idx, fmap in enumerate(feature_maps):
			nearest_fmaps = torch.index_select(self.feature_maps[idx], 0, indices)
			# min() because kappa=1 in the paper
			s_map, _ = torch.min(torch.linalg.norm(nearest_fmaps - fmap, dim=1), 0, keepdims=True)
			scaled_s_map += torch.nn.functional.interpolate(
				s_map.unsqueeze(0), size=(self.image_size,self.image_size), mode='bilinear'
			)

		scaled_s_map = self.blur(scaled_s_map)
		
		return z_score, scaled_s_map

	def get_parameters(self):
		return super().get_parameters({
			"k": self.k,
		})

class PaDiM(KNNExtractor):
	def __init__(
		self,
		d_reduced: int = 100,
		backbone_name: str = "resnet18",
	):
		super().__init__(
			backbone_name=backbone_name,
			out_indices=(1,2,3),
		)
		self.image_size = 224
		self.d_reduced = d_reduced # your RAM will thank you
		self.epsilon = 0.04 # cov regularization
		self.patch_lib = []
		self.resize = None

	def fit(self, train_dl):
		for sample, _ in tqdm(train_dl, **get_tqdm_params()):
			feature_maps = self(sample)
			if self.resize is None:
				largest_fmap_size = feature_maps[0].shape[-2:]
				self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
			resized_maps = [self.resize(fmap) for fmap in feature_maps]
			self.patch_lib.append(torch.cat(resized_maps, 1))
		self.patch_lib = torch.cat(self.patch_lib, 0)

		# random projection
		if self.patch_lib.shape[1] > self.d_reduced:
			print(f"   PaDiM: (randomly) reducing {self.patch_lib.shape[1]} dimensions to {self.d_reduced}.")
			self.r_indices = torch.randperm(self.patch_lib.shape[1])[:self.d_reduced]
			self.patch_lib_reduced = self.patch_lib[:,self.r_indices,...]
		else:
			print("   PaDiM: d_reduced is higher than the actual number of dimensions, copying self.patch_lib ...")
			self.patch_lib_reduced = self.patch_lib

		# calcs
		self.means = torch.mean(self.patch_lib, dim=0, keepdim=True)
		self.means_reduced = self.means[:,self.r_indices,...]
		x_ = self.patch_lib_reduced - self.means_reduced

		# cov calc
		self.E = torch.einsum(
			'abkl,bckl->ackl',
			x_.permute([1,0,2,3]), # transpose first two dims
			x_,
		) * 1/(self.patch_lib.shape[0]-1)
		self.E += self.epsilon * torch.eye(self.d_reduced).unsqueeze(-1).unsqueeze(-1)
		self.E_inv = torch.linalg.inv(self.E.permute([2,3,0,1])).permute([2,3,0,1])

	def predict(self, sample):
		feature_maps = self(sample)
		resized_maps = [self.resize(fmap) for fmap in feature_maps]
		fmap = torch.cat(resized_maps, 1)

		# reduce
		x_ = fmap[:,self.r_indices,...] - self.means_reduced

		left = torch.einsum('abkl,bckl->ackl', x_, self.E_inv)
		s_map = torch.sqrt(torch.einsum('abkl,abkl->akl', left, x_))
		scaled_s_map = torch.nn.functional.interpolate(
			s_map.unsqueeze(0), size=(self.image_size,self.image_size), mode='bilinear'
		)

		return torch.max(s_map), scaled_s_map[0, ...]

	def get_parameters(self):
		return super().get_parameters({
			"d_reduced": self.d_reduced,
			"epsilon": self.epsilon,
		})


class PatchCore(KNNExtractor):
	def __init__(
		self,
		f_coreset: float = 0.01, # fraction the number of training samples
		backbone_name : str = "resnet18",
		coreset_eps: float = 0.90, # sparse projection parameter
	):
		super().__init__(
			backbone_name=backbone_name,
			out_indices=(2,3),
		)
		self.f_coreset = f_coreset
		self.coreset_eps = coreset_eps
		self.image_size = 224
		self.average = torch.nn.AvgPool2d(3, stride=1)
		self.blur = GaussianBlur(4)
		self.n_reweight = 3

		self.patch_lib = []
		self.resize = None

	def fit(self, train_dl):
		for sample, _ in tqdm(train_dl, **get_tqdm_params()):
			feature_maps = self(sample)

			if self.resize is None:
				largest_fmap_size = feature_maps[0].shape[-2:]
				self.resize = torch.nn.AdaptiveAvgPool2d(largest_fmap_size)
			resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
			patch = torch.cat(resized_maps, 1)
			patch = patch.reshape(patch.shape[1], -1).T

			self.patch_lib.append(patch)

		self.patch_lib = torch.cat(self.patch_lib, 0)

		if self.f_coreset < 1:
			self.coreset_idx = get_coreset_idx_randomp(
				self.patch_lib,
				n=int(self.f_coreset * self.patch_lib.shape[0]),
				eps=self.coreset_eps,
			)
			self.patch_lib = self.patch_lib[self.coreset_idx]

	def predict(self, sample):		
		feature_maps = self(sample)
		resized_maps = [self.resize(self.average(fmap)) for fmap in feature_maps]
		patch = torch.cat(resized_maps, 1)
		patch = patch.reshape(patch.shape[1], -1).T

		dist = torch.cdist(patch, self.patch_lib)
		min_val, min_idx = torch.min(dist, dim=1)
		s_idx = torch.argmax(min_val)
		s_star = torch.max(min_val)

		# reweighting
		m_test = patch[s_idx].unsqueeze(0) # anomalous patch
		m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0) # closest neighbour
		w_dist = torch.cdist(m_star, self.patch_lib) # find knn to m_star pt.1
		_, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False) # pt.2
		# equation 7 from the paper
		m_star_knn = torch.linalg.norm(m_test-self.patch_lib[nn_idx[0,1:]], dim=1)
		# Softmax normalization trick as in transformers.
		# As the patch vectors grow larger, their norm might differ a lot.
		# exp(norm) can give infinities.
		D = torch.sqrt(torch.tensor(patch.shape[1]))
		w = 1-(torch.exp(s_star/D)/(torch.sum(torch.exp(m_star_knn/D))))
		s = w*s_star

		# segmentation map
		s_map = min_val.view(1,1,*feature_maps[0].shape[-2:])
		s_map = torch.nn.functional.interpolate(
			s_map, size=(self.image_size,self.image_size), mode='bilinear'
		)
		s_map = self.blur(s_map)

		return s, s_map


	def get_parameters(self):
		return super().get_parameters({
			"f_coreset": self.f_coreset,
			"n_reweight": self.n_reweight,
		})
