## BeCLR: Bregman Divergence Improves Contrastive Learning of Visual Representations
This is the official PyTorch implementation of the BeCLR paper:
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
optional arguments:
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 100]
```
