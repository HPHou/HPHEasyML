import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X,y = iris['data'][:,2:], iris['target']

clf = LogisticRegression(random_state=42,solver='lbfgs',multi_class='multinomial').fit(X, y)

print(clf.score(X, y))
