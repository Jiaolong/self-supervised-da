Self-supervised Domain Adaptation
=================================

Repository for the arXiv paper ["Self-supervised Domain adaptation for Computer Vision Tasks"](https://arxiv.org/abs/1907.10915).

```
@article{self-supervised-da:2019,
  title={Self-supervised Domain Adaptation for Computer Vision Tasks},
  author={Jiaolong, Xu and Liang, Xiao and Antonio M. López},
  journal={arXiv preprint arXiv:1907.10915},
  year={2019}
}
```

## Requirements

- python3.5+

- pytorch 1.0+

## Prepare dataset

Please find the PACS dataset from [this link](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)

The directories of the dataset are as following:

```
.
├── datasets
│   └── PACS
│       └── kfold
│           ├── art_painting
│           ├── cartoon
│           ├── photo
│           └── sketch
```

## Running experiments

The configuration files for each experiment can be found at `config/` folder.

For example:

```shell
python3 main.py --config configs/rotate_pacs_photo.yaml
```

To reproduce the results, running each experiment for three repeatitions with random seeds from `100`, `200` and `300`.

## Results

| Method  | art paint.| cartoon | sketches | photo | Avg. |
| --------|-----------|---------|----------|-------|------|
| SRC[1]  | 77.85  |74.86 |67.74 |95.73 |79.05|
| JigGen[1]  | 84.88 |81.07 |79.05 |97.96 |85.74|
|Ours(SRC) |79.33 |76.75 |64.40 |96.39 |79.22|
| Ours(Jigsaw) |84.93 |83.85 |69.04 |93.92 |82.94|
|Ours(Rot) |89.35 |84.14 |79.54 |98.24 |87.82|

## Acknowledgement

Thanks for the open source of [JigGen](https://github.com/fmcarlucci/JigenDG) for reference implementation!

## References

[1] F. M. Carlucci, A. D’Innocente, S. Bucci, B. Caputo, and T. Tommasi. Domain generalization by solving jigsaw puzzles. In CVPR, 2019.
