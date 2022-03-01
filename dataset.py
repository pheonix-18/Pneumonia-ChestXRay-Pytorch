import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PneumoniaDataset(Dataset):
    def __init__(self, path, transforms=None, limit=10):
        self.path = path
        from random import shuffle, seed;
        seed(10)
        self.X = glob.glob(path + '/*/*.jpeg')
        shuffle(self.X)
        # Load limited or else Load all
        print(limit)
        if limit > 0:
            print("Limiting Samples")
            self.X = self.X[:limit]
        self.Y = [int(label == 'PNEUMONIA') for label in [x.split('/')[-2] for x in self.X]]
        self.transforms = transforms
        print("Loaded Test Samples : ", len(self.X))
    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        img = cv2.imread(self.X[ix])
        label = self.Y[ix]
        return img, label

    def collate_fn(self, batch):
        _imgs, _classes = list(zip(*batch))
        if self.transforms:
            imgs = [self.transforms(img).unsqueeze(0) for img in _imgs]
        classes = [torch.tensor(cls).unsqueeze(0) for cls in _classes]
        imgs = torch.cat(imgs, dim=0).float().to(device)
        classes = torch.cat(classes).float().to(device)
        return imgs, classes

