import numpy as np 
import matplotlib.pyplot as plt 


xmin, xmax = -3, 3
xx = np.linspace(xmin, xmax, 100)

plt.plot([xmin, 0, 0, xmax], [1, 1, 0, 0],color='r',label='0-1 loss')
plt.plot(xx, np.where(xx < 1, 1 - xx, 0), color='g', label='Hinge loss')
plt.plot(xx, -np.minimum(xx, 0), color='b',label='Perceptron loss')
plt.plot(xx, np.log2(1 + np.exp(-xx)), color='y',label='Log loss')
plt.plot(xx, np.where(xx < 1, 1 - xx, 0) ** 2, color='orange', label="Squared hinge loss")

plt.ylim((0, 6))
plt.xlim(-3,3)
plt.legend(loc="upper right")
plt.xlabel(r"Decision function $f(x)$")
plt.ylabel("$L(y=1, f(x))$")
plt.grid()
plt.show()