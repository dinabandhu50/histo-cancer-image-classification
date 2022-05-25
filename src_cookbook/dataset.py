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

transform = transforms.Compose([transforms.ToTensor()])

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, df, resize=None, augmentations=None):
        self.path2data = os.path.join(root_dir,"train")
        self.df = df
        # get full path
        self.full_filenames = [os.path.join(self.path2data, f + ".tif") for f in self.df.id]
        # get labels
        self.labels = [t for t in self.df.label]
        # resize
        self.resize = resize
        # obtain transforms
        self.augmentations = augmentations

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = image.convert("RGB")
        target = self.labels[idx]
                
         # resize if needed
        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]),resample=Image.Resampling.BILINEAR)

        # convert to numpy 
        image = np.array(image)
        target = np.array(target)

        if self.augmentations is not None:
            augments = self.augmentations(image=image)
            image = augments["image"]

        # pytorch expects CHW instead of HWC
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        # target = torch.tensor(target).reshape(-1)

        return {
            "x" : torch.tensor(image,dtype=torch.float),
            "y" : torch.tensor(target,dtype=torch.long),
        }


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, df, resize=None, augmentations=None):
        self.path2data = os.path.join(root_dir,"test")
        self.df = df
        # get full path
        self.full_filenames = [os.path.join(self.path2data, f + ".tif") for f in self.df.id]
        # get labels
        self.labels = [t for t in self.df.label]
        # resize
        self.resize = resize
        # obtain transforms
        self.augmentations = augmentations

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])
        image = image.convert("RGB")
        target = self.labels[idx]
                
         # resize if needed
        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]),resample=Image.Resampling.BILINEAR)

        # convert to numpy 
        image = np.array(image)
        target = np.array(target)

        if self.augmentations is not None:
            augments = self.augmentations(image=image)
            image = augments["image"]

        # pytorch expects CHW instead of HWC
        image = np.transpose(image,(2,0,1)).astype(np.float32)
        # target = torch.tensor(target).reshape(-1)

        return {
            "x" : torch.tensor(image,dtype=torch.float),
            "y" : torch.tensor(target,dtype=torch.long),
        }


class HistoCancerDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=transforms.Compose([transforms.ToTensor()]), data_type="train"):
        # path to data folder
        data_type = data_type.lower()
        path2data = os.path.join(data_dir, data_type)
        # # path to csv labels file
        path2csvlabels = os.path.join(data_dir,f"{data_type}_labels.csv")
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

    start_time = time.time()

    input_dir = os.path.join(config.ROOT_DIR,"inputs")
    df_fold = pd.read_csv(os.path.join(input_dir,"train_folds.csv"))
    df_train = df_fold[df_fold.kfold == 0]
    df_valid = df_fold[df_fold.kfold != 0]

    resize = 64
    transform = transforms.Compose([transforms.ToTensor()])
    # transform = None
    train_data = TrainDataset(root_dir= input_dir,df = df_train,resize=(resize,resize), augmentations=None)
    valid_data = TrainDataset(root_dir= input_dir,df = df_valid,resize=(resize,resize), augmentations=None)

    img , target = train_data[9]["x"], train_data[9]["y"]
    img_valid , target_valid = valid_data[9]["x"], valid_data[9]["y"]

    # print(img, target)
    print(img.shape, target.shape)
    print(img.dtype, target.dtype)
    print("target: ",target)
    print("Dataset length: ",len(train_data))
    print(f"Duration of code execution: {time.time() - start_time}")

    # plt.figure()
    # plt.imshow(img)
    # plt.show()

    plt.figure()
    plt.imshow(np.transpose(img.numpy()*255,[1,2,0]).astype(np.uint8))
    # plt.show()

    plt.figure()
    plt.imshow(np.transpose(img_valid.numpy()*255,[1,2,0]).astype(np.uint8))
    plt.show()



    # print('-'*60)
    # ## Using HistCancerDataset
    # start_time = time.time()
    # data = HistoCancerDataset(data_dir=os.path.join(config.ROOT_DIR,"inputs"),)
    # img, target = data[9]

    # # print(img, target)
    # print(img.shape, target.shape)
    # print(img.dtype, target.dtype)
    # print("target: ",target)
    # print("Dataset length: ",len(train_data))
    # print(f"Duration of code execution: {time.time() - start_time}")

