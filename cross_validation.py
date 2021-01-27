# # %%
# import seaborn as sns
# import pandas as pd
# from sklearn import model_selection

# # %%
# df = pd.read_csv("data/train.csv")
# # %%
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
# if __name__ == '__main__':
#     df = pd.read_csv('data/train.csv')
#     # new column kfold and fill it with -1
#     df['kfold'] = -1

#     # sample
#     df = df.sample(frac=1).reset_index(drop=True)
#     # assuming the data has a targets column
#     y = df.target.value

#     kf = model_selection.StratifiedKFold(n_splits=5)

#     for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
#         df.loc[val_, 'kfold'] = fold
#     df.to_csv('train_folds.csv', index=False)
# #%%
# b = sns.countplot(x='quality', data=df)
# b.set_xlabel("quality", fontsize=20)
# b.set_ylabel("count", fontsize=20)


# %%
# stratified-kfold for regression
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import datasets


def create_folds(data):
    data['kfold'] = -1
    data.sample(frac=1).reset_index(drop=True)

    # calculate number of bins using sturge's rule
    # number of bins = 1 + np.log2(N) N is the data length
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # bin targets
    data.loc[:, 'bins'] = pd.cut(data['target'], bins=num_bins, labels=False)

    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the kfold  column
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    # drop the bins column
    data = data.drop('bins', axis=1)

    # return df with folds
    return data


if __name__ == '__main__':
    # sample dataset with 15000 stuff
    X, y = datasets.make_regression(
        n_samples=15000, n_features=100, n_targets=1)

    df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
    df.loc[:, 'target'] = y

    # create folds
    df = create_folds(df)
