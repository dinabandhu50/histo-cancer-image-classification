# write the training code here
import os
import torch
from torch.utils.data import random_split

import config
from dataset import HistoCancerDataset

# load data
histo_dataset = HistoCancerDataset(data_dir=os.path.join(config.ROOT_DIR,"inputs"))

# split the data
len_histo = len(histo_dataset)
len_train = int(0.8*len_histo)
len_valid = len_histo - len_train

train_dataset, valid_dataset = random_split(histo_dataset,[len_train,len_valid])

print(f"train dataset length: {len(train_dataset)}")
print(f"valid dataset length: {len(valid_dataset)}")

print("-"*20,"checking dim","-"*20)



