import os
import config
import pandas as pd

from sklearn.model_selection import KFold


def create_folds(df):
    df.loc[:,"kfold"] = -1

    kf = KFold(n_splits=5)
    for fold, (trn, val_idx) in enumerate(kf.split(X=df.id,y=df.label)):
        df.loc[val_idx,"kfold"] = fold
    df.to_csv(os.path.join(config.ROOT_DIR, "inputs","train_folds.csv"),index=False)
    return df


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(config.ROOT_DIR, "inputs","train_labels.csv"))    
    print(df.head())

    df2 = create_folds(df)
    print(df2.head())