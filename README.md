# ResNet-Implementation

"Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun

Paper: https://arxiv.org/pdf/1512.03385.pdf

## Architecture

<img src="./images/architecture.jpg" alt="ResNet Architecture" style="width:100%;">

## ResNet34

<img src="./images/resnet34.png" alt="ResNet34" style="width:100%;">


## Sad Story

No GPU, No Training.

## Info

Run script below to checkout the model informations

```sh
python info.py
```

## Usage

Before running the script, place your data directory location for both train and test data in `root_dir="{DIR}"` here at [dataloader.py](./dataloader/dataloader.py) or datasets from [torchvision.datasets](https://pytorch.org/vision/0.8/datasets.html)

```sh
python train.py --epochs 100 --num_layers 34 --num_classes 1000
```