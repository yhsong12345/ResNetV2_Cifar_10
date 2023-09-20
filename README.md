# ResNetV2_Cifar_10

This is the practice for the implementation of ResNetV2.  Models are trained with Cifar 10.

## Prerequisites
- Pytorch 2.0.1
- Python 3.11.4
- Window 11
- conda 23.7.4

## Training
```
# GPU training
python train.py -m ResnetV2-110 -e 200 -lr 0.01 -b 128 -s 32 -d outputs
```

## Testing
```
python test.py -m ResnetV2-110 -e 200 -lr 0.01 -b 128 -s 32 -d outputs
```

## Result (Accuracy)

Pretrained model should be downloaded if you click the name of Model.

| Model             | Acc.        |
| ----------------- | ----------- |
| [ResNet20]()          | 91.48     |
| [ResNet32]()          | 91.65%      |
| [ResNet44]()         | 91.68%      |
| [ResNet56]()          | 91.54%      |
| [Plain20]()          | 90.3%      |
| [Plain32]()         | 89.38%      |

## Plot
Plots are in the plots folder.
