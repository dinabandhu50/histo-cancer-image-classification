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
        # return {
            # "image" : torch.tensor(image,dtype=torch.float),
            # "targets" : torch.tensor(target,dtype=torch.long),
        # }

if __name__ == '__main__':
    
    import config
    import matplotlib.pyplot as plt
    import time

    start_time = time.time()
    data = HistoCancerDataset(data_dir=os.path.join(config.ROOT_DIR,"inputs"),data_type="test")
    img , target = data[9]
    # print(img, target)
    print(img.shape, target.shape)
    print("target: ",target)
    print("Dataset length: ",len(data))
    print(f"Duration of code execution: {time.time() - start_time}")
    # plt.figure()
    # plt.imshow(img)
    # plt.show()

    # plt.figure()
    # plt.imshow(img.numpy().astype(np.uint8))
    # plt.show()
