# write the training code here
import os
import config
import time
import copy

import numpy as np
import pandas as pd

import albumentations
import torch

from sklearn import metrics
from sklearn.model_selection import train_test_split

import dataset
import engine
from model import get_model




if __name__ == '__main__':
    # location of train.csv and train_png folder
    data_path = os.path.join(config.ROOT_DIR,"inputs")

    # cuda/cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # let's train for 5 epochs
    epochs = 5

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

    # mean and std values of RGB channels for imagenet dataset
    # we use these pre-calculated values when we use weights
    # from imagenet.
    # when we do not use imagenet weights, we use the mean and
    # standard deviation values of the original dataset
    # please note that this is a separate calculation
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # albumentations is an image augmentation library
    # that allows you do to many different types of image
    # augmentations. here, i am using only normalization
    # notice always_apply=True. we always want to apply
    # normalization
    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ])

    # instead of using kfold, i am using train_test_split
    # with a fixed random state
    train_images, valid_images, train_targets, valid_targets = train_test_split(images, targets, stratify=targets, random_state=42)

    # hyper-parameters
    BATCH_SIZE = 64
    NUM_WORKERS = 6
    RESIZE = 128
    # LR = 3e-4
    LR = 1e-5 

    # fetch the ClassificationDataset class
    train_dataset = dataset.ClassificationDataset(
    image_paths=train_images,
    targets=train_targets,
    resize=(RESIZE, RESIZE),
    augmentations=aug,
    )
    # torch dataloader creates batches of data
    # from classification dataset class
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # same for validation data
    valid_dataset = dataset.ClassificationDataset(
    image_paths=valid_images,
    targets=valid_targets,
    resize=(RESIZE, RESIZE),
    augmentations=aug,
    )
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # simple Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # lr scheduler
    

    # infos
    print(f"using device: {device}")
    print(f"pretrained model: {model_name}")
    print(f"Epoch Size: {epochs}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Re-Size: {RESIZE}")

    # train and print auc score for all epochs
    for epoch in range(epochs):
        start_time = time.time()
        engine.train(train_loader, model, optimizer, device=device)
        # evaluate train
        train_preds, train_targets = engine.evaluate(train_loader, model, device=device)
        train_roc_auc = metrics.roc_auc_score(train_targets, train_preds)
        # evaluate valid
        valid_preds, valid_targets = engine.evaluate(valid_loader, model, device=device)
        valid_roc_auc = metrics.roc_auc_score(valid_targets, valid_preds)

        print(f"Epoch={epoch}, Train ROC-AUC={train_roc_auc: 0.4f}, Valid ROC-AUC={valid_roc_auc: 0.4f}, time={((time.time() - start_time))/60} mins ")
