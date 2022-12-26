# Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces (Demo Code)

[![Unit Test](https://github.com/p16i/drsa-demo/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/p16i/drsa-demo/actions/workflows/pytest.yml)


We provide two Jupyter notebooks (in `./notebooks`):

| Name (`.ipynb`) | Description | Related Content |
|:--------|:---------|:-------|
| `disentangled-explanation` | Disentanged explanations from PRCA and DRSA for class `basketball` from [VGG16-TV][vgg16-tv]| Fig. 2 |
| `toy-example` | Comparison of PCA, DRSA, PRCA and DRSA on a synthetic distribution | Fig. 4 and Supplement Note F|

*Remark:* Make sure that `PYTHONPATH` includes `$(pwd)/cxai` when starting a Jupyter instance.


## Setup

We use Python version 3.8.6 or higher. We provide a list of necessary dependcies in `requirements.txt`. One can install them via

```
pip install -r requirements.txt
```

Please run the unit test command below to check that necessary functionalities work.

```
pytest tests/*
```
*Remark:* the command above approximately take 8 minutes to run.

[vgg16-tv]: https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html