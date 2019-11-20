
# Label, in PyTorch
This reposity is focus on finding some way to solve the label problem in supervised learning, such as noisy label, similar label and so on. The purpose is to form a solution to help solve the industrial or agricultural label problem. There are three part:
1. Label Represent
2. Label Regulation
3. Label Clean(Noisy label)
If some way can combine ,I will use the master branch to experience. But if some papers is diffcult to combine to master branch ,I will create some branch named paper. Currently, some experiments are carried out on the Cifar dataset, if you want to train your own dataset, you can refer the cifar list to create yourself dataset. 

### Table of Contents
- <a href='#Folder_Structure'>Folder</a>
- <a href='#Environment'>Installation</a>
- <a href='#Datasets'>Datasets</a>
- <a href='#Noisy_Label'>NoisyLabel</a>
- <a href='#Represent_Label'>RepresentLabel</a>
- <a href='#Regulate_Label'>RegulateLabel</a>
- <a href='#Autor'>Autor</a>
- <a href='#References'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Fold-Structure
The fold structure as follow:

- config.py
- loss.py
- main.py
- data/
	- __init__.py
 	- dataset.py
	- augment.py
- model/
	- __init__.py
	- BasicModule.py
	- Resnet.py
	- Resnext.py
- utils/
	- __init__.py
	- LabelGeneration.py
	- utils.py
	- Visualizer.py
- tools/
	- train.py
	- eval.py
	- test.py
- result/
- checkpoints/
	

## Environment
- pytorch 0.4.1
- python3+
- visdom 
	- for real-time loss visualization during training!
	```Shell
	pip install visdom
	```
	- Start the server (probably in a screen or tmux)
	```Shell
	python visdom
	```
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).

## Datasets
- Cifar10 or Cifar100: Download and get the dataset, if you want to create the noisy label,you can use the the nois_cifar10.py. The dataset and noisy_cifar10.py is provided [Baidu DRIVE-raw.tar.gz](https://pan.baidu.com/s/1CcbndLcw3Gc6ZqhB28buNg),passward:nlnu,then put cifar10/ in the data directory.


## NoisyLabel

**1.NLNL: Negative Learning for Noisy Label**
if you want to learn some information,you can watch the paper [NLNL]( 

### Training
- In the Label fold:

```Shell
bash tools/SelNLPL_train.sh
```
or
'''Shell
python main.py train
'''
- Note:
  * For training, default NVIDIA GPU.
  * if you use the SelNLPL_train.sh, you can set the parameters in this file.(see tools/SelNLPL_train.sh) 
  * if you use python main.py train, you can set the parameter in config.py set the default parameters.(see 'config.py`) 

### Test
- To test a trained network:

```Shell
bash tools/SelNLPL_test.sh
```
you should to change the load_model_path for yourself checkpoint (see tools/elNLPL_test.sh)


## RepresentLabel


## RegulateLabel


## Authors
* [**JavierHuang**](https://github.com/JaryHuang)
* []

## References
- [SSD: Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325)
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Generalized Intersection over Union](https://arxiv.org/abs/1902.09630)
- [SSD: Single Shot MultiBox Object Detector, in PyTorch](https://github.com/amdegroot/ssd.pytorch)
- [mmdet](https://github.com/open-mmlab/mmdetection)
- [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch)

