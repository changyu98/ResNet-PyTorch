### CIFAR

To run on Imagenet, place your `train` and `test` directories in `data`. 

Example commands: 
```bash
# Evaluate small ResNet on CPU
python main.py data -e -a resnet18 --pretrained 
```
```bash
# Evaluate medium ResNet on GPU
python main.py data -e -a resnet18 --pretrained --gpu 0 --batch-size 128
```
```bash
# Evaluate ResNet-50 for comparison
python main.py data -e -a resnet50 --pretrained --gpu 0
```

### Result

| Model structure | Top-1 error | Top-5 error |
| --------------- |:-----------:|:-----------:|
|  resnet18       | 30.24       | 10.92       |
|  resnet34       | 26.70       | 8.58        |
|  resnet50       | 23.85       | 7.13        |
|  resnet101      | 22.63       | 6.44        |
|  resnet152      | 21.69       | 5.94        |

### References

 - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

