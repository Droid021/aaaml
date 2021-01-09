# t-SNE MNIST
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn import manifold

# %matplotlib inline

# %%
data = datasets.fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True
)
# %%
images, targets = data
targets = targets.astype(int)
# %%
# reshape and visulize images with matplotlib
img = images[1, :].reshape(28, 28)
plt.imshow(img, cmap='grey')
# %%
tsne = manifold.TSNE(n_components=2, random_state=42)
transformed_data = tsne.fit_transform(images[:3000, :])

# pandas dataframe
tsne_df = pd.DataFrame(
    np.column_stack((transformed_data, targets[:3000])),
    columns=['x', 'y', 'targets']
)

tsne_df.loc[:, 'targets'] = tsne_df.targets.astype(int)

print(tsne_df.head())

# %%
# plot with seaborn and matplotlib
grid = sns.FacetGrid(tsne_df, hue='targets', size=8)

grid.map(plt.scatter, 'x', 'y').add_legend()
