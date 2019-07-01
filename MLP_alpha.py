"""
A comparison of different values for regularization parameter 'aplha' on synthetic datasets. The plot shows that different alpha yield different desicion functions.

Alpha is a parameter for regularization term, aka penalty term, that combats overfitting by constraining the size of the weights, Increasing alpha may fix high varience (a sign of overfitting) by encouraging smaller weights, resulting in a decision boundary plot that appears with lesser curvatures. Similarly, decreasing alpha may fix high bias(a sign of underfiting) by encouraging larger weights, potentially resulting in a more complicated decision boudary.
"""
print(__doc__)

import numpy as np 
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons,make_classification
from sklearn.neural_network import MLPClassifier

alphas = np.logspace(-5,1,3)
names = ['aplha = ' + str(i) for i in alphas]
classifiers = [MLPClassifier(hidden_layer_sizes=(10,10),alpha=i,random_state=2) for i in alphas]

X, y = make_moons(n_samples=200,noise=0.1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)

x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min,x_max,0.2),np.arange(y_min, y_max,0.2))

figure = plt.figure(figsize=(17,4))
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000','#0000FF'])
ax = plt.subplot(1,4, 1)
ax.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap=cm_bright)
ax.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap=cm_bright,alpha=0.6)
ax.set_xlim(xx.min(),xx.max())
ax.set_ylim(yy.min(),yy.max())
ax.set_xticks(())
ax.set_yticks(())

i = 2
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1,4,i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    Z = clf.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1].reshape(xx.shape)
    ax.contourf(xx, yy, Z ,cmap=cm, alpha=0.8)
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=cm_bright, edgecolors='black',s=25)
    ax.scatter(X_test[:,0],X_test[:,1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='black',s=25)

    ax.set_xlim(xx.min(),xx.max())
    ax.set_ylim(yy.min(),yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - 0.3, yy.min() + 0.3, ('%.2f' % score).lstrip('0'),size=15,horizontalalignment='right')

    i += 1

figure.subplots_adjust(left=0.02,right=0.98)
plt.show()