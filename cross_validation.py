# %%
# kfold cross-validation
import pandas as pd
from sklearn import model_selection
from sklearn import datasets
# %%
if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")

    # create a new column and fill it wit -1
    df['kfold'] = -1
    print(df.head)

    # randomize the rows
    df = df.sample(frac=1).reset_index(drop=True)

    # init the kfold class from model selection
    kf = model_selection.KFold(n_splits=5)

    # fill the fold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
        print(df.loc[val_, 'kfold'])
    
    df.to_csv('data/train_folds.csv', index=False)
