# Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces (Demo Code)

[![TPAMI](https://img.shields.io/badge/DOI-10.1109/TPAMI.2024.3388275-0173b3.svg)][paper]
[![arXiv](https://img.shields.io/badge/arXiv-2212.14855-b31b1b.svg)](https://arxiv.org/abs/2212.14855)
[![Unit Test](https://github.com/p16i/drsa-demo/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/p16i/drsa-demo/actions/workflows/pytest.yml)

The repository contains demo code for our paper
> [*P Chormai, J Herrmann, KR Müller, G Montavon, "Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces", IEEE TPAMI 2024*][paper].


<p align="center">
    <img width="700px" src="https://private-user-images.githubusercontent.com/1214890/326379598-abecb911-2103-437f-bd32-58627bfc3f7c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQzODEzNjIsIm5iZiI6MTcxNDM4MTA2MiwicGF0aCI6Ii8xMjE0ODkwLzMyNjM3OTU5OC1hYmVjYjkxMS0yMTAzLTQzN2YtYmQzMi01ODYyN2JmYzNmN2MucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MDQyOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDA0MjlUMDg1NzQyWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9YTNmNDNjNDI1MjI5ZDlhY2M1NTIwZDg5YTlmMWVlZmQ4YmUyOTI4Mjg3ZDY1YzRkZWM1Nzk5OWQ1YjlhNmQ0NSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmYWN0b3JfaWQ9MCZrZXlfaWQ9MCZyZXBvX2lkPTAifQ.4PTBuZ-hYoQU14yUM2lW5YL3_WU8R_Ol6IoU4jGuxt4"/>
</p>
<br/>

The repository includes two Jupyter notebooks, namely
1. `./notebooks/demo.ipynb` demonstrates our  disentangled explanation framework. More specifically, the notebook shows how to obtain disentanged explanations from PRCA and DRSA for class `basketball` using activation and LRP context vectors from [`VGG16-TV`][vgg16-tv] at `Conv4_3`. The demonstration reproduces Fig. 1 in the main paper.
    <p align="center">
        <img width="600px" src="https://private-user-images.githubusercontent.com/1214890/326376616-493b91b1-9184-4210-987f-867cecdd660b.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQzODA3MTUsIm5iZiI6MTcxNDM4MDQxNSwicGF0aCI6Ii8xMjE0ODkwLzMyNjM3NjYxNi00OTNiOTFiMS05MTg0LTQyMTAtOTg3Zi04NjdjZWNkZDY2MGIucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MDQyOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDA0MjlUMDg0NjU1WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9OTBjNzFlYTkzM2JkZjk5ZDU2OGJiMWYxZGMzNTEyYWRjZjk0Mzk5OTVjZmQ2YzdkODA2Zjk0NTkwNzg0YzVmZCZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmYWN0b3JfaWQ9MCZrZXlfaWQ9MCZyZXBvX2lkPTAifQ.iTvM4yFoesQQy0ZKToSuUwwPMwBxhQptAudkFshOmKY">
    </p>
2. `./notebooks/lrp-nfnet.ipynb` demonstrates our LRP implementation for NFNet-F0. It reproduces heatmaps similar to the ones in Fig. D.2 in Supplementary Note D.
    <p align="center">
        <img width="500px" src="https://private-user-images.githubusercontent.com/1214890/326376682-2995bda3-6b5b-4009-aceb-a1375a458d4f.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQzODA3MTUsIm5iZiI6MTcxNDM4MDQxNSwicGF0aCI6Ii8xMjE0ODkwLzMyNjM3NjY4Mi0yOTk1YmRhMy02YjViLTQwMDktYWNlYi1hMTM3NWE0NThkNGYucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MDQyOSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDA0MjlUMDg0NjU1WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9OGNlZWQxNGJlYzlhYWY3MmEyZWY5MTE5YjUzNWZmYmQwNjM2ZTAxOTVjZGFjZWRiN2ZhYjczMTY1OWQ4ODcxMyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmYWN0b3JfaWQ9MCZrZXlfaWQ9MCZyZXBvX2lkPTAifQ.EMAQdGtA1J06-g_m8MBUrE3LsRWqjXGT8mkD9EtqBfU">
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

