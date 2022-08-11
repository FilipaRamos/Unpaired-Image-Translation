import os
import glob
import torch
import random

from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

"""
    Prepare images
    Data augmentation commented due to the outcomes of the described experiments
"""
class Transform():
    def __init__(self, img_size=256):
        self.transform = {
            'train': transforms.Compose([
                #transforms.RandomVerticalFlip(),
                #transforms.RandomHorizontalFlip(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]),
            'test': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])}
    def __call__(self, img, mode):
        return self.transform[mode](img)

class MonetDataset(Dataset):
    def __init__(self, photo_paths, monet_paths, transform, mode='train'):
        self.photo_paths = photo_paths
        self.monet_paths = monet_paths
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return min([len(self.photo_paths), len(self.monet_paths)])
        
    def __getitem__(self, index):        
        photo_path = self.photo_paths[index]
        monet_path = self.monet_paths[index]
        photo = Image.open(photo_path)
        monet = Image.open(monet_path)
        
        photo_t = self.transform(photo, self.mode)
        monet_t = self.transform(monet, self.mode)

        return photo_t, monet_t

class MonetDataLoader():
    def __init__(self, photo_path, monet_path, batch_size, transform, mode='train', seed=0):
        self.photo_path = photo_path
        self.monet_path = monet_path
        self.batch_size = batch_size
        self.transform = transform
        self.mode = mode
        self.seed = seed

    def prepare_data(self):
        self.photo_paths = glob.glob(os.path.join(self.photo_path, '*.jpg'))
        self.monet_paths = glob.glob(os.path.join(self.monet_path, '*.jpg'))

    def train_dataloader(self):
        torch.manual_seed(self.seed)
        self.train_dataset = MonetDataset(self.photo_paths, self.monet_paths, self.transform, self.mode)
        
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=4
                         )