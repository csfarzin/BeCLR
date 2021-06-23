from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset


def load_brain(path):
    f = np.load(path)
    x, y = f['x_train'], f['y_train']
    f.close()
    
    x = x.astype(np.float32)
    #x = np.transpose(x.data, (0, 3, 1, 2))
    x = np.divide(x, 255.)
    y = y.astype(np.int32)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.85, random_state=5)
    print('Brain samples shapes x_train: {}, x_test: {}'.format(x_train.shape, x_test.shape))
    
    return x_train, y_train, x_test, y_test
    
class BrainDataset(Dataset):

    def __init__(self, path='./data/brain_rgb.npz', train=True, transform=transforms.ToTensor()):
        self.x_train, self.y_train, self.x_test, self.y_test = load_brain(path)
        self.train = train
        self.transform = transform

    def __len__(self):
        if self.train:
            return self.x_train.shape[0]
        else:
            return self.x_test.shape[0]

    def __getitem__(self, idx):
        
        if self.train:
            img = self.x_train
            label = self.y_train
        else:
            img = self.x_test
            label = self.y_test
        
        if self.transform:
            img = self.trasform(img)
        
        
        return img, torch.tensor(self.y[idx])