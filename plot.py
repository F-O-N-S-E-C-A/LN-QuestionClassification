from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def PLOT(corpus, train_X):
    pca = PCA(n_components = 4).fit(train_X.toarray())
    X_pca = pca.transform(train_X.toarray())

    labels = [i for i in corpus[0]]

    sns.scatterplot(X_pca[:,0], X_pca[:, 1],  hue=labels, legend='full',palette="Set2")
    plt.show()

