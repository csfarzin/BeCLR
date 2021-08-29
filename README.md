## BeCLR: Bregman Divergence Improves Contrastive Learning of Visual Representations
<p align="center">
  <img src="https://github.com/csfarzin/BeCLR/blob/master/media/BeCLR_Arch.jpg" width="800"/>
</p>

<p align="justify">
<strong>Abstract</strong>: Deep divergence learning aims to measure the divergence of data points using neural networks and tune the deep neural networks for the relevant tasks. In this paper, we propose deep Bregman divergences for contrastive learning of visual representation. We aim to enhance contrastive loss used in self-supervised by learning functional Bregman divergences. In contrast to the conventional contrastive learning methods which are solely based on divergences between single points, our framework can capture the divergence between distributions which improves the quality of learned representation. Our setup exhibits desirable properties such as invariance to viewpoints, deformation, and intra-class variations. Our experiments show that learned representations beside learning divergences, one can considerably improve the baseline on several tasks such as classification and object detection. 
</p>

----
###This is the official PyTorch implementation of the BeCLR paper:
```
@Article{


}
```



## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```
- pyyaml
```
conda install -c conda-forge pyyaml
```
- tensorboard
```
conda install -c conda-forge tensorboard
```

## Dataset
All datasets will be downloaded into `data` directory by `PyTorch` automatically except ImageNet.
To download ImageNet dataset follow the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

## Training
### Train BeCLR
To train a model on datasets [cifar10, cifar100, stl10, skin] run main.py 
```
python main.py --batch_size 1024 --epochs 1000 
optional arguments:
--feature_dim                 Feature dim for latent vector [default value is 128]
--temperature                 Temperature used in softmax [default value is 0.5]
--k_nn                        Top k most similar images used to predict the label [default value is 200]
--k_subs                      Number of subnetworks corresponding to a set of affine functionals
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 500]
-dataset-name                 Name of the dataset (e.g., cifar10, cifar100, stl10, ...)
```
To train a model on ImageNet dataset run main_imgnet.py 
```
python main.py --batch_size 512 --epochs 200
optional arguments:
--feature_dim                 Feature dim for latent vector [default value is 128]
--temperature                 Temperature used in softmax [default value is 0.1]
--k_nn                        Top k most similar images used to predict the label [default value is 200]
--k_subs                      Number of subnetworks corresponding to a set of affine functionals
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 200]
```
### Linear Evaluation
With a pre-trained model, to train a supervised linear classifier on frozen weights, run:
```
python linear.py --batch_size 1024 --epochs 100 
arguments:
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 100]
```
### Fine-Tuning
Fine-tune on the previously learned representations with a subset of the dataset. 
```
python fine_tuning.py
arguments:
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 100]
--dataset-name                Name of the dataset (e.g., cifar10, cifar100, ...)
--base_model                  Base Network Architecture (resnet18, resnet50)
--model_path                  Path to pretrained model ("/path/to/checkpoint/resnet50.pth")
```
### Object Detection
This part is adopted from MoCo object detection implementation. Please follow this [instruction](https://github.com/facebookresearch/moco/tree/master/detection) to run this part.
    
## Logging and TensorBoard
To view results in TensorBoard, run:
```
tensorboard --logdir runs
```
