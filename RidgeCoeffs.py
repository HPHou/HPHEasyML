import matplotlib.pyplot as plt 
import numpy as np 

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

clf = Ridge()

X, y, w = make_regression(n_samples=10,n_features=10,coef=True,
                        noise=50,random_state=42,bias=.5)

coefs = []
errors = []
alphas = np.logspace(-6, 6, 200)

for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X,y)
    coefs.append(clf.coef_)
    errors.append(mean_squared_error(clf.coef_, w))

plt.subplot(121)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weight')
plt.title('Ridge coefficients as a function of the regularization')

plt.subplot(122)
ax = plt.gca()
ax.plot(alphas,errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Coefficient error as a function of the regularization')


plt.show()
