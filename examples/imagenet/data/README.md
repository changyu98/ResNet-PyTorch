### Imagenet

To run on Imagenet, place your `train` and `val` directories in `data`. 

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
