import os
import config
import time

import numpy as np
import pandas as pd
import albumentations


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset

import engine
from model import get_model

from sklearn import metrics

class HP:
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 2
    batch_size_train = 128
    batch_size_valid = 256
    num_workers = 6
    resize = 64
    lr = 3e-4


def train_fold(fold, df):

    input_dir = os.path.join(config.ROOT_DIR,"inputs")
    df_train = df[df.kfold == fold]
    df_valid = df[df.kfold != fold]

    # define augmentations
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ])

    train_data = TrainDataset(root_dir= input_dir,df = df_train, resize= (HP.resize,HP.resize), augmentations=aug)
    valid_data = TrainDataset(root_dir= input_dir,df = df_valid, resize= (HP.resize,HP.resize), augmentations=aug)

    # img , target = train_data[9]["x"], train_data[9]["y"]
    # img_valid , target_valid = valid_data[9]["x"], valid_data[9]["y"]

    train_loader = DataLoader(train_data, batch_size = HP.batch_size_train, num_workers = HP.num_workers)
    valid_loader = DataLoader(valid_data, batch_size = HP.batch_size_train, num_workers = HP.num_workers)

    model_name = 'resnet18'
    model = get_model(model_name=model_name)
    # move model to device
    model.to(HP.device)

    # loss function 
    criterion = nn.BCEWithLogitsLoss()
    # simple Adam optimizer
    optimizer = torch.optim.Adam([
        { 'params': model.layer4.parameters(), 'lr': HP.lr/3},
        { 'params': model.layer3.parameters(), 'lr': HP.lr/9},
    ], lr=HP.lr)

    # unfreeze the last layers
    unfreeze_layers = [model.layer3, model.layer4]
    for layer in unfreeze_layers:
        for param in layer.parameters():
            param.requires_grad = True

    # infos
    print(f"using device: {HP.device}")
    print(f"pretrained model: {model_name}")
    print(f"Epoch Size: {HP.epochs}")
    print(f"Train batch Size: {HP.batch_size_train}")
    print(f"Valid batch Size: {HP.batch_size_valid}")
    print(f"Re-Size: {HP.resize}")
    print(f"len(data_loader) ={len(train_loader)}, len(valid_loader)={len(valid_loader)}")
    print("----------start training---------\n")
    
    # train and print auc score for all epochs
    for epoch in range(HP.epochs):
        start_time = time.time()
        engine.train(train_data, train_loader, model, criterion, optimizer, device=HP.device)
        # evaluate train
        train_preds, train_targets, avg_train_loss = engine.evaluate(train_data, train_loader, model, criterion, device=HP.device)
        train_roc_auc = metrics.roc_auc_score(train_targets, train_preds)
        # evaluate valid
        valid_preds, valid_targets, avg_valid_loss = engine.evaluate(valid_data, valid_loader, model, criterion, device=HP.device)
        valid_roc_auc = metrics.roc_auc_score(valid_targets, valid_preds)

        print(f"Epoch={epoch}, Train ROC-AUC={train_roc_auc: 0.4f}, Valid ROC-AUC={valid_roc_auc: 0.4f}, time={((time.time() - start_time))/60} mins ")

    print("\n----------------Done---------------")


if __name__ == '__main__':
    FOLDS = 2
    input_dir = os.path.join(config.ROOT_DIR,"inputs")
    df_fold = pd.read_csv(os.path.join(input_dir,"train_folds.csv"))

    for fold in range(FOLDS):
        print(f"FOLD - {fold}")
        train_fold(fold=fold, df=df_fold)
        # break
