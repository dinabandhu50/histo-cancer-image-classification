import os
import config
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import albumentations
import torch
import torch.nn as nn


from sklearn import metrics
from sklearn.model_selection import train_test_split

import math
import dataset
from model import get_model


def find_lr(model, loss_fn, optimizer, train_loader, init_value=1e-8, final_value=10.0):
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    count = 0
    for data in train_loader:
        batch_num += 1
        # remember we have image and target in our dataset class
        inputs = data["image"]
        targets = data["targets"]

        # move inputs/targets to cuda/cpu device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)

        # inputs, labels = data
        # inputs, labels = inputs, labels
        optimizer.zero_grad()
        outputs = model(inputs)
        # print("op shape",outputs.shape)

        # loss = loss_fn(outputs, targets)
        # loss = loss_fn(outputs, targets.view(-1,1))
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))

        # Crash out if loss explodes
        if batch_num > 1 and loss > 4 * best_loss:
            # return log_lrs[10:-5], losses[10:-5]
            return log_lrs, losses
        # Record the best loss
        if loss < best_loss or batch_num == 1:
            best_loss = loss
        # Store the values
        temp_loss = loss.detach().cpu().numpy().tolist()
        print(f"count: {count}, loss: {temp_loss}")
        losses.append(temp_loss)
        # print("loss after",loss.detach().cpu().numpy().tolist())

        log_lrs.append(math.log10(lr))
        # Do the backward pass and optimize
        loss.backward()
        optimizer.step()
        # Update the lr for the next step and store
        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
        count += 1
        # if count>10:
        #     break

    # return log_lrs[10:-5], losses[10:-5]
    return log_lrs, losses



if __name__ == '__main__':
    # location of train.csv and train_png folder
    data_path = os.path.join(config.ROOT_DIR,"inputs")
    # cuda/cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the dataframe
    df = pd.read_csv(os.path.join(data_path,"train_labels.csv"))
    # a list of all image location
    images = [os.path.join(data_path,"train",f"{item}.tif") for item in df.id.values.tolist()]
    # get the target list
    targets = df.label.values.tolist()

    # get the model
    # model_name = "alexnet"
    # model_name = "resnet18"
    model_name = "mobilenet_v2"

    model = get_model(model_name=model_name,pretrained=True)
    # move model to device
    model.to(device)

    # augment
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ])
    
    # instead of using kfold, i am using train_test_split with a fixed random state
    train_images, valid_images, train_targets, valid_targets = train_test_split(images, targets, stratify=targets, random_state=42)

    # hyper-parameters
    # batch_size = 32
    batch_size = 64
    num_workers = 6
    resize = 128

    # fetch the ClassificationDataset class
    train_dataset = dataset.ClassificationDataset(
                                image_paths=train_images,
                                targets=train_targets,
                                resize=(resize, resize),
                                augmentations=aug,
                                )
    # torch dataloader creates batches of data
    # from classification dataset class
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # simple Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # function call
    logs,losses = find_lr(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn, 
        train_loader=train_loader
    )

    # plot
    plt.figure(figsize=(7,10))
    plt.plot(logs,losses)
    plt.show()
    found_lr = 1e-5