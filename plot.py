from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def PLOT(corpus, train_X, train_Y):
    pca = PCA(n_components = 9).fit(train_X.toarray())
    X_pca = pca.transform(train_X.toarray())

    labels = [corpus[0][i] for i in train_Y]
    sns.scatterplot(X_pca[:,0], X_pca[:, 1],  hue=labels, legend='full',palette="Set2")
    plt.show()
