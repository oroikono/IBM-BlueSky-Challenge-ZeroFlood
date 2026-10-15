# ZeroFlood: IBM BlueSky Challenge Project

ZeroFlood explores flood hazard mapping with satellite imagery and geospatial foundation models. This repository contains the IBM BlueSky Challenge project, with training and inference notebooks.

The related paper, **“ZeroFlood: Flood Hazard Mapping from Single-Modality SAR Using Geo-Foundation Models,”** by Hyeongkyun Kim and Orestis Oikonomou, studies flood hazard mapping from synthetic aperture radar (SAR) imagery. For the implementation linked by the latest paper, use [khyeongkyun/zeroflood](https://github.com/khyeongkyun/zeroflood).

[Paper, arXiv v2](https://arxiv.org/abs/2510.23364v2) · [Paper implementation](https://github.com/khyeongkyun/zeroflood) · [Citation](#citation)

![Demo](./asset/demo.gif)

[Detailed Information](https://huggingface.co/spaces/ibm-esa-geospatial/challenge/discussions/2)

## Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oroikono/IBM-BlueSky-Challenge-ZeroFlood/blob/main/training.ipynb)

## Inference

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oroikono/IBM-BlueSky-Challenge-ZeroFlood/blob/main/inference.ipynb)

## Citation

If you use this implementation or build on the paper's method, please cite:

```bibtex
@misc{kim2025zerofloodv2,
  title = {{ZeroFlood}: Flood Hazard Mapping from Single-Modality {SAR} Using Geo-Foundation Models},
  author = {Kim, Hyeongkyun and Oikonomou, Orestis},
  year = {2025},
  eprint = {2510.23364},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  doi = {10.48550/arXiv.2510.23364},
  url = {https://arxiv.org/abs/2510.23364v2},
  note = {Version 2, revised 30 March 2026; first submitted in 2025}
}
```
