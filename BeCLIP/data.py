import logging
import os
from PIL import Image
import random

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from clip import tokenize

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t"):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = tokenize([str(self.captions[idx])])[0]
        return images, texts


def get_csv_dataloader(args, preprocess_fn):
    input_filename = args.data_train
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return dataloader

def get_imagenet_dataloader(args, config):
    # get data for evaluation
    mean= [0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((config['image_resolution'], config['image_resolution'])),
                                    transforms.Normalize(mean, std)])
    val_dir = os.path.join(args.data_val, 'val')

    test_data = ImageFolder(val_dir, transform)

    test_loader = DataLoader(
        test_data, batch_size=1024, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    return test_loader

class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,batch_first=False):
        self.batch_first = batch_first
    
    def __call__(self,batch):
        imgs = []
        caps = []
        for item in batch:
            img, txts = item
            txt = random.choice(txts)
            imgs.append(img.unsqueeze(0))
            caps.append(txt)
            
        imgs = torch.cat(imgs,dim=0)
        targets = tokenize(caps)
        return imgs, targets

def get_coco_dataloader(args, preprocess_fn):
    data_path = args.data_train
    train_path = os.path.join(data_path, 'train2014/')
    train_ann_path = os.path.join(data_path, 'annotations/captions_train2014.json')
    val_path = os.path.join(data_path, 'val2014/')
    val_ann_path = os.path.join(data_path, 'annotations/captions_val2014.json')

    train_set = torchvision.datasets.CocoCaptions(root=train_path, annFile=train_ann_path,
                                                  transform=preprocess_fn)
    val_set = torchvision.datasets.CocoCaptions(root=val_path, annFile=val_ann_path,
                                                  transform=preprocess_fn)
    
    pretrain_set = torch.utils.data.ConcatDataset((train_set, val_set))
    print(len(pretrain_set))

    sampler = DistributedSampler(pretrain_set) if args.distributed else None

    train_loader = DataLoader(pretrain_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True,
                              sampler=sampler,
                              drop_last=True,
                              collate_fn=CapsCollate(batch_first=True))
    return train_loader

    