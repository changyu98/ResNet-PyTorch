# The complete implementation of the CIFAR10 dataset in the ResNet paper

The official implementation of ImageNet is included, but the parameters of this dataset cannot be extended to cifar10 because it brings more of the appearance of overfitting of neural networks.If you categorize cifar10 directly with the official implementation, the accuracy of the paper is far from being achieved.To this end, I implemented the following code based on the description in the [original paper](http://xxx.itp.ac.cn/abs/1512.03385).

Details about the models are below: 

|     *Method*      |*# Params*|*Error (paper).*|*Error (ours).*|*Pretrained?*|
|:-----------------:|:--------:|:--------------:|:-------------:|:-----------:|
|    `resnet20`     |  0.27M   |    8.75%       |     8.02%     |      ×      |
|    `resnet32`     |  0.46M   |    7.51%       |     7.35%     |      ×      |
|    `resnet44`     |  0.66M   |    7.17%       |     6.70%     |      ×      |
|    `resnet56`     |  0.85M   |    6.97%       |     6.60%     |      ×      |
|    `resnet110`    |   1.7M   |6.43%(6.61±0.16)|     6.26%     |      ×      |
|    `resnet1202`   |  19.4M   |    7.93%       |       ——      |      ×      |
 

### Train
```text
python main.py data -a resnet20 --gpu 0 
```

### Evaluate
```text
python main.py data -a resnet20 --gpu 0 -e --resume model_best.pth
```

### download pre-trained weights

- [resnet20.pth](https://raw.githubusercontent.com/Lornatang/ResNet/master/examples/cifar/pretrained_weight/resnet20.pth)
- [resnet32.pth](https://raw.githubusercontent.com/Lornatang/ResNet/master/examples/cifar/pretrained_weight/resnet32.pth)
- [resnet44.pth](https://raw.githubusercontent.com/Lornatang/ResNet/master/examples/cifar/pretrained_weight/resnet44.pth)
- [resnet56.pth](https://raw.githubusercontent.com/Lornatang/ResNet/master/examples/cifar/pretrained_weight/resnet56.pth)
- [resnet110.pth](https://raw.githubusercontent.com/Lornatang/ResNet/master/examples/cifar/pretrained_weight/resnet110.pth)
