import itertools
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
from scipy import linalg 
from sklearn import mixture

color_iter = itertools.cycle(['navy','c','cornflowerblue','gold','darkorange'])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1 ,1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle,color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9, 5)
    plt.ylim(-3, 6)
    plt.title(title)

n_samples = 500
np.random.seed(0)
C = np.array([[0, -0.1], [1.7, 0.4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
            0.7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture')

dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type='full').fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,'Bayesian Gaussian Mixture with a Dirchlet process prior')

plt.show()