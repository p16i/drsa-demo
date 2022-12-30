# Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces (Demo Code)

[![Unit Test](https://github.com/p16i/drsa-demo/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/p16i/drsa-demo/actions/workflows/pytest.yml)


We provide two Jupyter notebooks, namely
1. `./notebooks/lrp-nfnet.ipynb` demonstrates our LRP implementation for NFNet-F0. It reproduces Fig. D.2 in Supplementary Note D.
2. `./notebooks/demo.ipynb` demonstrates our  disentangled explanation framework. More specifically, the notebook shows how to obtain disentanged explanations from PRCA and DRSA for class `basketball` using activation and LRP context vectors from [`VGG16-TV`][vgg16-tv] at `Conv4_3`. The demonstration reproduces Fig. 2 in the main paper.

*Remark:* Make sure that `PYTHONPATH` includes `$(pwd)/cxai` when starting a Jupyter instance. Or, start the instance using `PYTHONPATH=$(pwd)/cxai jupyter notebook`.

## Setup

We use Python version 3.8.6. Necessary dependencies can be installed via
```
pip install -r requirements.txt
```

Please run the unit test command below to check that necessary functionalities work.

```
# testing important functions (approximately 3 minutes on CPUs)
make fast-test

# test all functions (approximately 6 minutes on Tesla V100)
make test
```

[vgg16-tv]: https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html