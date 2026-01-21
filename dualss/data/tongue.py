import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import pandas as pd
import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

class TongueBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 train_or_val="train",
                 use_aug = True,
                 dir_or_csv="dir",
                 all_in_memory = False,
                 fold_i = 0,
                 interpolation = 'bilinear'
                 ):

        self.data_root = data_root
        self.size = size
        self.is_train = True if train_or_val == "train" else False
        self.use_aug = use_aug
        self.from_dir = True if dir_or_csv == "dir" else False
        self.all_in_memory = all_in_memory


        if self.use_aug: 
            self.image_transforms = transforms.Compose([ transforms.RandomHorizontalFlip(),
                                                    transforms.RandomRotation(15, fill=0),
                                                    transforms.Resize((size, size)),
                                                    transforms.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=0.5, std=0.5)])
        else: 
            self.image_transforms = transforms.Compose([ transforms.Resize((size, size)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=0.5, std=0.5)])

        if self.from_dir:
            dir_list = os.listdir(data_root) 
            raw_len = len(dir_list)
            if self.is_train: 
                self.image_paths = dir_list[int(raw_len*0.05):int(raw_len*0.95)]
            else:
                self.image_paths = dir_list[:int(raw_len*0.05)]
                self.image_paths = self.image_paths + dir_list[int(raw_len*0.95):]
            self._length = len(self.image_paths)
            self.labels = self.image_paths
        else:
            category_num = 3
            fold_i = 0
            csv_dir = './Tongue-FLD/Diagnosis/'
            
            if self.is_train:
                csv_data = csv_dir + f'train_val_{category_num}_fold_{fold_i:02}.csv'
            else:
                csv_data = csv_dir + f'test_{category_num}_fold_{fold_i:02}.csv'
            data = pd.read_csv(csv_data)
            self.image_paths = list(data.iloc[:,0])
            self.column = data.columns.values.tolist()
            self.indicators = data.iloc[:,1:-1].values
            self.labels = F.one_hot(torch.tensor([int(l[-1]) for l in data.iloc[:,-1]])).float()

        self._length = len(self.image_paths)
        self.file_path = [os.path.join(self.data_root, l) for l in self.image_paths]

        if self.all_in_memory:
            self.image_in_memory = [Image.open(f).convert("RGB") for f in self.file_path]
            self.read_image = self.read_image_from_memory
        else:
            self.read_image = self.read_image_from_path


    def read_image_from_path(self, index):
        image = Image.open(self.file_path[index]).convert("RGB")
        return image

    def read_image_from_memory(self, index):
        image = self.image_in_memory[index]
        return image

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = { "image": self.image_transforms(self.read_image(i)),
                    "class_label": self.labels[i]}
        return example

DaTaRooT = "./Tongue-FLD/Tongue_Images/"

class TongueTrain(TongueBase):
    def __init__(self, **kwargs):
        super().__init__(train_or_val="train", use_aug=True, data_root=DaTaRooT, **kwargs)

class TongueValidation(TongueBase):
    def __init__(self, **kwargs):
        super().__init__(train_or_val="val", use_aug=False, data_root=DaTaRooT, **kwargs)

