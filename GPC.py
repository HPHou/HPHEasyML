import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

train_size = 50
rng = np.random.RandomState(42)
X = rng.uniform(0, 5, 100)
X = X[:, np.newaxis]
y = np.array(X[:, 0] > 2.5, dtype=int)

gp_fix = GaussianProcessClassifier(kernel=1.0 * RBF         (length_scale=1.0),optimizer=None)
gp_fix.fit(X[:train_size],y[:train_size])

gp_opt = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
gp_opt.fit(X[:train_size],y[:train_size])


print("Log Marginal Likehood (initial) : %.3f" % gp_fix.log_marginal_likelihood(gp_fix.kernel_.theta))
print("Log Marginal Likelihood (optimized): %.3f"
      % gp_opt.log_marginal_likelihood(gp_opt.kernel_.theta))

print("Accuracy: %.3f (initial) %.3f (optimized)"
      % (accuracy_score(y[:train_size], gp_fix.predict(X[:train_size])),
         accuracy_score(y[:train_size], gp_opt.predict(X[:train_size]))))
print("Log-loss: %.3f (initial) %.3f (optimized)"
      % (log_loss(y[:train_size], gp_fix.predict_proba(X[:train_size])[:, 1]),
         log_loss(y[:train_size], gp_opt.predict_proba(X[:train_size])[:, 1])))