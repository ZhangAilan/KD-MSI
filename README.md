## Prerequisite
```
imageio==2.9.0
numpy==1.21.2
opencv_python_headless==4.6.0.66
pandas==1.3.4
Pillow==10.1.0
pydensecrf==1.0rc2
torch==1.12.1+cu116
torchvision==0.11.2
```
## Dataset Preparation

### Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt, test.txt`, each file records the image names (XXX.png) in the change detection dataset. It also contains `train_label.txt`, which is the image level label of training data, please run generate_cls_label.py to generate this list.
