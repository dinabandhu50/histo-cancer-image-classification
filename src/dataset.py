# dataset classes here
import torch
import torchvision.transforms as transforms
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
    def __init__(self, image_paths, targets, resize=None, augmentations=None):
        """Image classification dataset for binary classification.

        Args:
            image_paths ([list]): list containing full path to images to be read
            targets ([list]): list containing target values corresponding to image 
            resize (optional): resize param, default is None.
            augmentations ([type], optional): albumentation image augmentations. Defaults to None.
            returns ([dict]): returns dictionary containing image and the label of that image as follow  
            {
                "x":image , # torch.tensor datatype
                "y":target, # torch.tensor datatype
            } 
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        target = self.targets[idx]

        # resize if needed
        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]),resample=Image.BILINEAR)
        
        # convert to numpy 
        image = np.array(image)
        target = np.array(target)

        if self.augmentations is not None:
            augments = self.augmentations(image=image)
            image = augments["image"]
        
        # pytorch expects CHW instead of HWC
        image = np.transpose(image,(2,0,1)).astype(np.float32)

        return {
            "image" : torch.tensor(image,dtype=torch.float),
            "targets" : torch.tensor(target,dtype=torch.long),
        }


class HistoCancerDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=transforms.Compose([transforms.ToTensor()])):
    # def __init__(self, data_dir, transform=None):
        # path to data folder
        path2data = os.path.join(data_dir,'train')
        # # path to csv labels file
        path2csvlabels = os.path.join(data_dir,"train_labels.csv")
        # read label file
        labels_df = pd.read_csv(path2csvlabels)
        # get full path
        self.full_filenames = [os.path.join(path2data, f + ".tif") for f in labels_df.id]
        # obtain labels
        self.labels = [t for t in labels_df.label]
        # obtain transforms
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self,idx):
        # read image
        image = Image.open(self.full_filenames[idx])
        # read target
        target = self.labels[idx]
        # image to np.array
        image = np.array(image)
        
        # using ToTensor No need of transposing
        # # transpose image to torch format [B,C,H,W] 
        # image = np.transpose(image,[2,0,1])
        
        # do the transformations
        if self.transform is not None:
            image = self.transform(image)

        # don't forget to transform.ToTensor() for image
        return image, torch.tensor(target).reshape(-1)


if __name__ == '__main__':
    
    import config
    import matplotlib.pyplot as plt
    import time

    # start_time = time.time()
    # data = HistoCancerDataset(data_dir=os.path.join(config.ROOT_DIR,"inputs"))
    # img , target = data[9]
    # # print(img, target)
    # print(img.shape, target.shape)
    # print("target: ",target)
    # print("Dataset length: ",len(data))
    # print(f"Duration of code execution: {time.time() - start_time}")
    # # plt.figure()
    # # plt.imshow(img)
    # # plt.show()

    ## ---------------------------------------------------------------------------------
    start_time = time.time()
    data_path = os.path.join(config.ROOT_DIR,"inputs")
    df = pd.read_csv(os.path.join(config.ROOT_DIR,"inputs","train_labels.csv"))
    images = df.id.values.tolist()
    images = [os.path.join(data_path,"train",img+".tif") for img in images]
    targets = df.label.values
    data = ClassificationDataset(image_paths=images,targets=targets)
    # idx = 9
    # img, target = data[idx]["image"], data[idx]["targets"]
    # print(img, target)
    # print(img.shape,torch.min(img),torch.max(img))
    print(f"Duration of code execution: {time.time() - start_time}")

    # plt.figure()
    # plt.imshow(img.numpy().astype(np.uint8))
    # plt.show()
    
