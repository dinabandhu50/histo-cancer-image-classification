# dataset classes here
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

import os
import numpy as np
import pandas as pd

from utils import seed_everything

seed_everything(seed=42)

# some time corrupted images won't load, so use the following line
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset(Dataset):
    def __init__(self, image_paths, targets, augmentations=None):
        """Image classification dataset for binary classification.

        Args:
            image_paths ([list]): list containing full path to images to be read
            targets ([list]): list containing target values corresponding to image 
            augmentations ([type], optional): albumentation image augmentations. Defaults to None.
            returns ([dict]): returns dictionary containing image and the label of that image as follow  
            {
                "x":image , # torch.tensor datatype
                "y":target, # torch.tensor datatype
            } 
        """
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx])
        target = Image.open(self.targets[idx])

        # convert to numpy 
        image = np.array(image)
        target = np.array(target)

        if self.augmentations is not None:
            augments = self.augmentations(image=image)
            image = augments["image"]
        
        # pytorch expects CHW instead of HWC
        image = np.transpose(image,(2,0,1)).astype(np.float32)

        return {
            "x" : torch.tensor(image,dtype=torch.float),
            "y" : torch.tensor(target,dtype=torch.long),
        }

        