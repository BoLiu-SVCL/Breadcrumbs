# Breadcrumb: Adversarial Class-balanced Sampling for Long-tailed Recognition

## Overview
This is the author's pytorch implementation for the paper "Breadcrumb: Adversarial Class-balanced Sampling for Long-tailed Recognition". This will reproduce the Breadcrumb performance on ImageNet-LT with ResNet10.

The model is designed to train on eight (8) Titan Xp (12GB memory each). Please adjust the batch size (or even learning rate) accordingly, if the GPU setting is different.

Dataloader and sampler inherit from OTLR.

## Requirements
* [Python](https://python.org/) (version 3.7.6 tested)
* [PyTorch](https://pytorch.org/) (version 1.5.1 tested)

## Data Preparation
- First, please download the [ImageNet_2014](http://image-net.org/index).

- Next, change the `data_root` in `train.py`, and `eval.py` accordingly.

- The data splits are provided in the codes.

## Getting Started (Training & Testing)
- training:
```
sh train.sh
```
- testing:
```
sh eval.sh
```
