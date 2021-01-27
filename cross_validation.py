# %%

import pandas as pd
from sklearn import model_selection
# %%
# kfold cross-validation
# if __name__ == "__main__":
#     df = pd.read_csv("data/train.csv")

#     # create a new column and fill it wit -1
#     df['kfold'] = -1
#     print(df.head)

#     # randomize the rows
#     df = df.sample(frac=1).reset_index(drop=True)

#     # init the kfold class from model selection
#     kf = model_selection.KFold(n_splits=5)

#     # fill the fold column
#     for fold, (trn_, val_) in enumerate(kf.split(X=df)):
#         df.loc[val_, 'kfold'] = fold
#         print(df.loc[val_, 'kfold'])

#     df.to_csv('data/train_folds.csv', index=False)

# %%
# stratified k-fold (maintains the ratio of labels in each fold)
if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    # new column kfold and fill it with -1
    df['kfold'] = -1

    # sample
    df = df.sample(frac=1).reset_index(drop=True)
    # assuming the data has a targets column
    y = df.target.value

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold
    df.to_csv('train_folds.csv', index=False)
