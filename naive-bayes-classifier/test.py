import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from matplotlib import cm

def plotcases(ax):
    plt.scatter(xdf['x1'],xdf['x2'],c=xdf['y'], cmap=cm.coolwarm, axes=ax, alpha=0.6, s=20, lw=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', left='off', right='off', top='off', bottom='off',
                   labelleft='off', labelright='off', labeltop='off', labelbottom='off')

# a function to draw decision boundary and colour the regions
def plotboundary(ax, Z):
    ax.pcolormesh(xx, yy, Z, cmap=cm.coolwarm, alpha=0.1)
    ax.contour(xx, yy, Z, linewidths=0.75, colors='k')

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.02, class_sep=0.5,
                           hypercube=True, shift=0.0, scale=0.5, shuffle=True, random_state=5)

xdf = pd.DataFrame(X, columns=['x1','x2'])
xdf['y'] = y


x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
nx, ny = 100, 100   # this sets the num of points in the mesh
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
np.linspace(y_min, y_max, ny))

gnb = GaussianNB()
gnb.fit(xdf[['x1','x2']], xdf['y'])

Z = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:,1].reshape(xx.shape)


ax = plt.subplot()
ax.title.set_text("Gaussian Naive Bayes")
plotboundary(ax, Z)
plotcases(ax)

plt.show()
