import os
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_skin(root='./data'):
    dataset_path = os.path.join(root, 'skin_rgb.npz')
    f = np.load(dataset_path)
    x, y = f['x_train'], f['y_train']
    f.close()
    
    x = x.astype(np.uint8)
    #x = np.transpose(x.data, (0, 3, 1, 2))
    #x = np.divide(x, 255.)
    y = list(y.astype(np.int32))
    x, y = shuffle(x, y, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=0)
    #print('Skin samples shapes x_train: {}, x_test: {}'.format(x_train.shape, x_test.shape))
    return x_train, y_train, x_test, y_test
    
class SKIN(Dataset):

    def __init__(self, root='./data', train=True, transform=transforms.ToTensor()):
        self.classes = ['MEL', 'NV', 'BBC', 'AKIEC', 'BKL', 'DF', 'VASC']
        
        x_train, y_train, x_test, y_test = load_skin(root)
        if train:
            self.data = x_train
            self.targets = y_train
        else:
            self.data = x_test
            self.targets = y_test
            
        self.train = train
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        img = self.data[idx]
        label = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        
        return img, label