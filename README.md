# ResNet

### Update (February 2, 2020)

This update allows you to use NVIDIA's Apex tool for accelerated training. By default choice `hybrid training precision` + `dynamic loss amplified` version, if you need to learn more and details about `apex` tools, please visit https://github.com/NVIDIA/apex.

### Overview
This repository contains an op-for-op PyTorch reimplementation of [ResNet](https://arxiv.org/pdf/1512.03385.pdf).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained ResNet models 
 * Use ResNet models for classification or feature extraction 

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an ResNet on your own dataset
 * Export ResNet models for production
 
### Table of contents
1. [About ResNet](#about-resnet)
2. [Installation](#installation)
3. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
4. [Contributing](#contributing) 

### About ResNet

If you're new to ResNets, here is an explanation straight from the official PyTorch implementation: 

Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.
The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

### Installation

Install from source:
```bash
git clone https://github.com/lornatang/ResNet
cd ResNet
python setup.py install
``` 

### Usage

#### Loading pretrained models

Load an resnet18 network:
```python
from resnet import ResNet
model = ResNet.from_name("resnet18")
```

Load a pretrained resnet18: 
```python
from resnet import ResNet
model = ResNet.from_pretrained("resnet18")
```

Details about the models are below: 

|  *Method*  |*#Params*|   *top-1 err*   |   *top-5 err*   |*Pretrained?*|
|:-----------|:-------:|:---------------:|:---------------:|:-----------:|
|`resnet18`  |  11.7M  |      27.88      |       —         |      √      |
|`resnet34`  |  21.8M  |24.52(24.61±0.42)| 7.46(7.58±0.18) |      √      |
|`resnet50`  |  25.6M  |      22.85      |      6.71       |      √      |
|`resnet101` |  44.6M  |      21.75      |      6.05       |      √      |
|`resnet156` |  60.2M  |      21.43      |      5.71       |      √      |

*Option B of resnet-18/34/50/101/152 only uses projections to increase dimensions.*

For results extending to the cifar10 dataset, see `examples/cifar`

#### Example: Classification

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`. 

```python
import json
import urllib

import torch
import torchvision.transforms as transforms
from PIL import Image

from resnet import ResNet

input_image = Image.open("img.jpg")

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

labels_map = json.load(open("labels_map.txt"))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify with ResNet
model = ResNet.from_pretrained("resnet18")
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():
    output = model(input_batch)
preds = torch.topk(output, k=5).indices.squeeze(0).tolist()

print("-----")
for idx in preds:
    label = labels_map[idx]
    prob = torch.softmax(output, dim=1)[0, idx].item()
    print("{:<75} ({:.2f}%)".format(label, prob * 100))
```

#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 