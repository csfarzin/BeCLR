import os
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset

################################################################################
################################################################################
################################################################################

def load_retina(path='./data/retina_rgb.npz'):
    f = np.load(path)
    x, y = f['x_train'], f['y_train']
    f.close()
    
    x = x.astype(np.float32)
    #x = np.transpose(x.data, (0, 3, 1, 2))
    x = np.divide(x, 255.)
    y = y.astype(np.int32)
    print('Retina samples', x.shape, y.shape)
    return x, y
    
class RetinaDataset(Dataset):

    def __init__(self, path='./data/retina_rgb.npz', transform=transforms.ToTensor()):
        self.x, self.y = load_retina(path)
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            images = self.transform(self.x[idx])
        
        return images, torch.tensor(self.y[idx])