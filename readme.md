# Industrial KNN-based Anomaly Detection

<img src="docs/example_anomaly_maps.png" width="500"/>

This repo aims to reproduce the results of the following KNN-based anomaly detection methods:

1. SPADE (Cohen et al. 2021) - knn in z-space and distance to feature maps
   ![spade schematic](docs/schematic_spade.png)
2. PaDiM* (Defard et al. 2020) - distance to multivariate Gaussian of feature maps
   ![padim schematic](docs/schematic_padim.png)
3. PatchCore (Roth et al. 2021) - knn distance to avgpooled feature maps
   ![patchcore schematic](docs/schematic_patchcore.png)

\* actually does not have any knn mechanism, but shares many things implementation-wise.

---

## Install

```shell
$ pipenv install -r requirements.txt
```
Note: I used torch cu11 wheels.

## Usage

CLI:
```shell
$ python indad/run.py METHOD [--dataset DATASET]
```
Results can be found under `./results/`.

Code example:
```python
from indad.model import SPADE

model = SPADE(k=42, backbone_name="hypernet")

# feed healthy dataset
model.fit(...)

# get predictions
img_lvl_anom_score, pxl_lvl_anom_score = model.predict(...)
```

### Custom datasets
<details>
  <summary> 👁️ </summary>

Check out one of the downloaded MVTec datasets.
Naming of images should correspond among folders.
Right now there is no support for no ground truth pixel masks.

```
📂datasets
 ┗ 📂your_custom_dataset
  ┣ 📂 ground_truth/defective
  ┃ ┣ 📂 defect_type_1
  ┃ ┗ 📂 defect_type_2
  ┣ 📂 test
  ┃ ┣ 📂 defect_type_1
  ┃ ┣ 📂 defect_type_2
  ┃ ┗ 📂 good
  ┗ 📂 train/good
```

```shell
$ python indad/run.py METHOD --dataset your_custom_dataset
```
</details>

---

## Results

📝 = paper, 👇 = this repo

### Image-level

| class      | SPADE 📝 | SPADE 👇 | PaDiM 📝 | PaDiM 👇| PatchCore 📝 | PatchCore 👇 |
|-----------:|:--------:|:--------:|:--------:|:-------:|:------------:|:------------:|
| bottle     | -        | 98.3     | 98.3     | 99.9    | **100.0**    | **100.0**    |
| cable      | -        | 88.1     | 96.7     | 87.8    | **99.5**     | 96.2         |
| capsule    | -        | 80.4     | **98.5** | 87.6    | 98.1         | 95.3         |
| carpet     | -        | 62.5     | 99.1     | **99.5**| 98.7         | 98.7         |
| grid       | -        | 25.6     | 97.3     | 95.5    | **98.2**     | 93.0         |
| hazelnut   | -        | 92.8     | 98.2     | 86.1    | **100.0**    | 100.0        |
| leather    | -        | 85.6     | 99.2     | 100.0   | **100.0**    | 100.0        |
| metal_nut  | -        | 78.6     | 97.2     | 97.6    | **100.0**    | 98.3         |
| pill       | -        | 78.8     | 95.7     | 92.7    | **96.6**     | 92.8         |
| screw      | -        | 66.1     | **98.5** | 79.6    | 98.1         | 96.7         |
| tile       | -        | 96.4     | 94.1     | **99.5**| 98.7         | 99.0         |
| toothbrush | -        | 83.9     | 98.8     | 94.7    | **100.0**    | 98.1         |
| transistor | -        | 89.4     | 97.5     | 95.0    | **100.0**    | 99.7         |
| wood       | -        | 85.3     | 94.7     | 99.4    | **99.2**     | 98.8         |
| zipper     | -        | 97.1     | 98.5     | 93.8    | **99.4**     | 98.4         |
| averages   | 85.5     | 80.6     | 97.5     | 93.9    | **99.1**     | 97.7         |

### Pixel-level

| class      | SPADE 📝 | SPADE 👇 | PaDiM 📝 | PaDiM 👇| PatchCore 📝 | PatchCore 👇 |
|-----------:|:--------:|:--------:|:--------:|:-------:|:------------:|:------------:|
| bottle     | 97.5     | 97.7     | 94.8     | 97.6    | **98.6**     | 97.8         | 
| cable      | 93.7     | 94.4     | 88.8     | 95.5    | **98.5**     | 97.4         | 
| capsule    | 97.6     | 98.7     | 93.5     | 98.1    | **98.9**     | 98.3         | 
| carpet     | 87.4     | 99.0     | 96.2     | 98.7    | **99.1**     | 98.3         | 
| grid       | 88.5     | 96.4     | 94.6     | 96.4    | **98.7**     | 96.7         | 
| hazelnut   | 98.4     | 98.4     | 92.6     | 97.3    | **98.7**     | 98.1         | 
| leather    | 97.2     | 99.1     | 97.8     | 98.6    | **99.3**     | 98.4         | 
| metal_nut  | **99.0** | 96.1     | 85.6     | 95.8    | 98.4         | 96.2         | 
| pill       | **99.1** | 93.5     | 92.7     | 94.4    | 97.6         | 98.7         | 
| screw      | 98.1     | 98.9     | 94.4     | 97.5    | **99.4**     | 98.4         | 
| tile       | **96.5** | 93.1     | 86.0     | 92.6    | 95.9         | 94.0         | 
| toothbrush | **98.9** | 98.9     | 93.1     | 98.5    | 98.7         | 98.1         | 
| transistor | **97.9** | 95.8     | 84.5     | 96.9    | 96.4         | 97.5         | 
| wood       | 94.1     | 94.5     | 91.1     | 92.9    | **95.1**     | 91.9         | 
| zipper     | 96.5     | 98.3     | 95.9     | 97.0    | **98.9**     | 97.6         | 
| averages   | 96.9     | 96.6     | 92.1     | 96.5    | **98.1**     | 97.2         |

__PatchCore-10 was used.__

### Hyperparams

The following parameters were used to calculate the results. 
They more or less correspond to the parameters used in the papers.

```yaml
spade:
  backbone: wide_resnet50_2
  k: 50
padim:
  backbone: wide_resnet50_2
  d_reduced: 250
  epsilon: 0.04
patchcore:
  backbone: wide_resnet50_2
  f_coreset: 0.1
  n_reweight: 3
```

---

## Progress

- [x] Datasets
- [x] Code skeleton
- [ ] Config files
- [x] CLI
- [x] Logging
- [x] SPADE
- [x] PADIM
- [x] PatchCore
- [x] Add custom dataset option
- [x] Add dataset progress bar
- [x] Add schematics
- [ ] Unit tests

## Design considerations

- Data is processed in single images to avoid batch statistics interference.
- I decided to implement greedy kcenter from scratch and there is room for improvement.
- `torch.nn.AdaptiveAvgPool2d` for feature map resizing, `torch.nn.functional.interpolate` for score map resizing.
- GPU is used for backbones and coreset selection. GPU coreset selection currently runs at:
  - 400-500 it/s @ float32 (RTX3080)
  - 1000+ it/s @ float16 (RTX3080)

---

## Acknowledgements

-  [hcw-00](https://github.com/hcw-00) for tipping `sklearn.random_projection.SparseRandomProjection`

## References

SPADE:
```bibtex
@misc{cohen2021subimage,
      title={Sub-Image Anomaly Detection with Deep Pyramid Correspondences}, 
      author={Niv Cohen and Yedid Hoshen},
      year={2021},
      eprint={2005.02357},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

PaDiM:
```bibtex
@misc{defard2020padim,
      title={PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization}, 
      author={Thomas Defard and Aleksandr Setkov and Angelique Loesch and Romaric Audigier},
      year={2020},
      eprint={2011.08785},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

PatchCore:
```bibtex
@misc{roth2021total,
      title={Towards Total Recall in Industrial Anomaly Detection}, 
      author={Karsten Roth and Latha Pemula and Joaquin Zepeda and Bernhard Schölkopf and Thomas Brox and Peter Gehler},
      year={2021},
      eprint={2106.08265},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
