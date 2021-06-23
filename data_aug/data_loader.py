from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import data_aug.datasets as utils
from data_aug.transformation import transform
import os

class CustomDataLoader():
    def __init__(self):
        pass
        
    def get_loader(self, dataset_name, batch_size, workers):
        
        if dataset_name == 'imagenet':
            # get test and train transformation
            mean= [0.485, 0.456, 0.406]
            std= [0.229, 0.224, 0.225]
            train_transform, test_transform = transform(224, mean, std)
            # data prepare
            root = '~/../DATA/ILSVRC/Data/CLS-LOC'
            traindir = os.path.join(root, 'train')
            valdir = os.path.join(root, 'val')
            
            train_data = ImageFolder(
            traindir,
            utils.TwoCropsTransform(train_transform))
            
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True, drop_last=True)

            memory_data = ImageFolder(
            traindir,
            utils.TwoCropsTransform(test_transform))
            memory_loader = DataLoader(
                memory_data, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)

            test_data = ImageFolder(
            valdir,
            utils.TwoCropsTransform(test_transform))
            test_loader = DataLoader(
                test_data, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)        

        elif dataset_name == 'cifar10':
            # get test and train transformation
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            train_transform, test_transform = transform(32, mean, std)
            # data prepare
            train_data = utils.CIFAR10Pair(
                root='data', train=True, transform=train_transform, download=True)
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True, drop_last=True)

            memory_data = utils.CIFAR10Pair(
                root='data', train=True, transform=test_transform, download=True)
            memory_loader = DataLoader(
                memory_data, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)

            test_data = utils.CIFAR10Pair(
                root='data', train=False, transform=test_transform, download=True)
            test_loader = DataLoader(
                test_data, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)
        
        elif dataset_name == 'cifar100':
            # get test and train transformation
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            train_transform, test_transform = transform(32, mean, std)
            # data prepare
            train_data = utils.CIFAR100Pair(
                root='data', train=True, transform=train_transform, download=True)
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True, drop_last=True)

            memory_data = utils.CIFAR100Pair(
                root='data', train=True, transform=test_transform, download=True)
            memory_loader = DataLoader(
                memory_data, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)

            test_data = utils.CIFAR100Pair(
                root='data', train=False, transform=test_transform, download=True)
            test_loader = DataLoader(
                test_data, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)
        
        elif dataset_name == 'stl10':
            # get test and train transformation
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2471, 0.2435, 0.2616]
            train_transform, test_transform = transform(96, mean, std)
            # data prepare
            train_data = utils.STL10Pair(
                root='data', split='train+unlabeled', transform=train_transform, download=True)
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True, drop_last=True)

            memory_data = utils.STL10Pair(
                root='data', split='train', transform=test_transform, download=True)
            memory_loader = DataLoader(
                memory_data, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)

            test_data = utils.STL10Pair(
                root='data', split='test', transform=test_transform, download=True)
            test_loader = DataLoader(
                test_data, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)
    
        elif dataset_name == 'skin':
            # get test and train transformation
            mean = [0.7630, 0.5463, 0.5706]
            std = [0.2471, 0.2435, 0.2616]
            train_transform, test_transform = transform(128, mean, std)
            # data prepare
            train_data = utils.SKINPair(
                root='data', train=True, transform=train_transform)
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True, drop_last=True)

            memory_data = utils.SKINPair(
                root='data', train=True, transform=test_transform)
            memory_loader = DataLoader(
                memory_data, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)

            test_data = utils.SKINPair(
                root='data', train=False, transform=test_transform)
            test_loader = DataLoader(
                test_data, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=True)
            
        
        return train_loader, memory_loader, test_loader
            
