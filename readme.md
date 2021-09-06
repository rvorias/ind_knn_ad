# Industrial KNN-based Anomaly Detection

⭐Now has streamlit support!⭐ Run `$ streamlit run streamlit_app.py`

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

## Usage

CLI:
```shell
$ python indad/run.py METHOD [--dataset DATASET]
```
Results can be found under `./results/`.

Code example:
```python
from indad.model import SPADE

model = SPADE(k=5, backbone_name="resnet18")

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

| class      | SPADE 📝 | SPADE 👇 | PaDiM 📝   | PaDiM 👇 | PatchCore 📝 | PatchCore 👇 |
|-----------:|:--------:|:--------:|:----------:|:--------:|:------------:|:------------:|
| bottle     | -        | 98.8     | 98.3       | 99.8     | ■**100.0**■  | ■**100.0**■  |
| cable      | -        | 76.5     | 96.7       | 93.3     | ■**99.5**■   | 96.2         |
| capsule    | -        | 84.6     | ■**98.5**■ | 88.3     | 98.1         | 95.3         |
| carpet     | -        | 84.3     | 99.1       | ■**99.4**| 98.7         | 98.7         |
| grid       | -        | 37.1     | 97.3       | 98.2     | ■**98.2**■   | 93.0         |
| hazelnut   | -        | 88.7     | 98.2       | 83.7     | ■**100.0**■  | 100.0        |
| leather    | -        | 97.1     | 99.2       | 99.9     | ■**100.0**■  | 100.0        |
| metal_nut  | -        | 74.6     | 97.2       | 99.4     | ■**100.0**■  | 98.3         |
| pill       | -        | 72.6     | 95.7       | 89.0     | ■**96.6**■   | 92.8         |
| screw      | -        | 53.1     | ■**98.5**■ | 83.0     | 98.1         | 96.7         |
| tile       | -        | 97.8     | 94.1       | 98.6     | 98.7         | ■**99.0**■   |
| toothbrush | -        | 89.4     | 98.8       | 97.2     | ■**100.0**■  | 98.1         |
| transistor | -        | 89.2     | 97.5       | 96.8     | ■**100.0**■  | 99.7         |
| wood       | -        | 98.3     | 94.7       | 98.9     | ■**99.2**■   | 98.8         |
| zipper     | -        | 96.7     | 98.5       | 89.5     | ■**99.4**■   | 98.4         |
| averages   | 85.5     | 82.6     | 97.5       | 94.3     | ■**99.1**■   | 97.7         |

### Pixel-level

| class      | SPADE 📝   | SPADE 👇   | PaDiM 📝 | PaDiM 👇| PatchCore 📝   | PatchCore 👇 |
|-----------:|:----------:|:----------:|:--------:|:-------:|:--------------:|:------------:|
| bottle     | 97.5       | 97.7       | 94.8     | 97.8    | ■**98.6**■     | 97.8         | 
| cable      | 93.7       | 94.3       | 88.8     | 96.1    | ■**98.5**■     | 97.4         | 
| capsule    | 97.6       | 98.6       | 93.5     | 98.3    | ■**98.9**■     | 98.3         | 
| carpet     | 87.4       | 99.0       | 96.2     | 98.6    | ■**99.1**■     | 98.3         | 
| grid       | 88.5       | 96.1       | 94.6     | 97.2    | ■**98.7**■     | 96.7         | 
| hazelnut   | 98.4       | 98.1       | 92.6     | 97.5    | ■**98.7**■     | 98.1         | 
| leather    | 97.2       | 99.2       | 97.8     | 98.7    | ■**99.3**■     | 98.4         | 
| metal_nut  | ■**99.0**■ | 96.1       | 85.6     | 96.5    | 98.4           | 96.2         | 
| pill       | ■**99.1**■ | 93.5       | 92.7     | 93.2    | 97.6           | 98.7         | 
| screw      | 98.1       | 98.9       | 94.4     | 97.8    | ■**99.4**■     | 98.4         | 
| tile       | ■**96.5**■ | 93.3       | 86.0     | 94.8    | 95.9           | 94.0         | 
| toothbrush | ■**98.9**■ | ■**98.9**■ | 93.1     | 98.3    | 98.7           | 98.1         | 
| transistor | ■**97.9**■ | 96.3       | 84.5     | 97.2    | 96.4           | 97.5         | 
| wood       | 94.1       | 94.4       | 91.1     | 93.6    | ■**95.1**■     | 91.9         | 
| zipper     | 96.5       | 98.2       | 95.9     | 97.4    | ■**98.9**■     | 97.6         | 
| averages   | 96.9       | 96.8       | 92.1     | 96.9    | ■**98.1**■     | 97.2         |

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

-  [hcw-00](https://github.com/hcw-00) for tipping `sklearn.random_projection.SparseRandomProjection`.
-  [h1day](https://github.com/h1day) for adding a custom range to the streamlit app.

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
