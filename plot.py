from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def PLOT_3D(train_X):
    NUMBER_OF_CLUSTERS = 5
    km = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        init='k-means++',
        max_iter=500)
    km.fit(train_X)
    # First: for every document we get its corresponding cluster
    clusters = km.predict(train_X)

    # We train the PCA on the dense version of the tf-idf.
    pca = PCA(n_components=3)
    two_dim = pca.fit_transform(train_X.todense())

    scatter_x = two_dim[:, 0] # first principle component
    scatter_y = two_dim[:, 1] # second principle component
    scatter_z = two_dim[:, 2] # third principle component

    plt.style.use('ggplot')

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')

    # color map for NUMBER_OF_CLUSTERS we have
    cmap = {0: 'green', 1: 'blue', 2: 'red', 3: 'yellow', 4: 'black'}

    # group by clusters and scatter plot every cluster
    # with a colour and a label
    for group in np.unique(clusters):
        ix = np.where(clusters == group)
        ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix] , c=cmap[group], label=group)


    ax.set_xlabel("0", fontsize=14)
    ax.set_ylabel("1", fontsize=14)
    ax.set_zlabel("2", fontsize=14)
    ax.legend()
    plt.show()

def PLOT_2D(train_X):
    NUMBER_OF_CLUSTERS = 5
    km = KMeans(
        n_clusters=NUMBER_OF_CLUSTERS,
        init='k-means++',
        max_iter=500)
    km.fit(train_X)
    # First: for every document we get its corresponding cluster
    clusters = km.predict(train_X)

    # We train the PCA on the dense version of the tf-idf.
    pca = PCA(n_components=2)
    two_dim = pca.fit_transform(train_X.todense())

    scatter_x = two_dim[:, 0] # first principle component
    scatter_y = two_dim[:, 1] # second principle component

    plt.style.use('ggplot')

    fig, ax = plt.subplots()
    fig.set_size_inches(20,10)

    # color map for NUMBER_OF_CLUSTERS we have
    cmap = {0: 'green', 1: 'blue', 2: 'red', 3: 'yellow', 4: 'black'}

    # group by clusters and scatter plot every cluster
    # with a colour and a label
    for group in np.unique(clusters):
        ix = np.where(clusters == group)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=cmap[group], label=group)

    ax.legend()
    plt.xlabel("0")
    plt.ylabel("1")
    plt.show()
