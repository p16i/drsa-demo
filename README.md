# Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces (Demo Code)

[![TPAMI](https://img.shields.io/badge/DOI-10.1109/TPAMI.2024.3388275-0173b3.svg)][paper]
[![arXiv](https://img.shields.io/badge/arXiv-2212.14855-b31b1b.svg)](https://arxiv.org/abs/2212.14855)
[![Unit Test](https://github.com/p16i/drsa-demo/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/p16i/drsa-demo/actions/workflows/pytest.yml)

The repository contains demo code for our paper
> [*P Chormai, J Herrmann, KR Müller, G Montavon, "Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces", IEEE TPAMI 2024*][paper].


<p align="center">
    <img width="700px" src="https://i.imgur.com/GtBJVkV.png"/>
</p>
<br/>

The repository includes two Jupyter notebooks, namely
1. `./notebooks/demo.ipynb` demonstrates our  disentangled explanation framework. More specifically, the notebook shows how to obtain disentanged explanations from PRCA and DRSA for class `basketball` using activation and LRP context vectors from [`VGG16-TV`][vgg16-tv] at `Conv4_3`. The demonstration reproduces Fig. 1 in the main paper.
    <p align="center">
        <img width="600px" src="https://i.imgur.com/I5JNr6l.png">
    </p>
2. `./notebooks/lrp-nfnet.ipynb` demonstrates our LRP implementation for NFNet-F0. It reproduces heatmaps similar to the ones in Fig. D.2 in Supplementary Note D.
    <p align="center">
        <img width="500px" src="https://i.imgur.com/WW5ajcZ.png">
    </p>

*Remark:* Make sure that `PYTHONPATH` includes `$(pwd)/cxai` when starting a Jupyter instance. Or, start the instance using `PYTHONPATH=$(pwd)/cxai jupyter notebook`.

If you find our demo code for your research, please consider citing our paper:

```
@ARTICLE{10497845,
  author={Chormai, Pattarawat and Herrmann, Jan and Müller, Klaus-Robert and Montavon, Grégoire},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces}, 
  year={2024},
  pages={1-18},
  doi={10.1109/TPAMI.2024.3388275}
}
```

# Setup

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
[paper]: https://ieeexplore.ieee.org/document/10497845

