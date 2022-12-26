# Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces (Demo Code)

[![Unit Test](https://github.com/p16i/drsa-demo/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/p16i/drsa-demo/actions/workflows/pytest.yml)


We provide a Jupyter notebook (`./notebooks/demo.ipynb`) that reproduces Fig. 2 in the main paper. More specically, the notebook demonstrates how to obtain disentanged explanations from PRCA and DRSA for class `basketball` using activation and LRP context vectors from [`VGG16-TV`][vgg16-tv] at `Conv4_3`.

*Remark:* Make sure that `PYTHONPATH` includes `$(pwd)/cxai` when starting a Jupyter instance. Or, start the instance using `PYTHONPATH=$(pwd)/cxai jupyter notebook`.


## Setup

We use Python version 3.8.6. Necessary dependencies can be installed via
```
pip install -r requirements.txt
```

Please run the unit test command below to check that necessary functionalities work.

```
pytest tests/*
```

*Remark:* the command above takes approximately 8 minutes to run.

[vgg16-tv]: https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html