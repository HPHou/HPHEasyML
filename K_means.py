import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

random_state = np.random.RandomState(42)

NUM_RUNS = 10
n_init_range = np.array([1, 5, 10, 15, 20])

n_samples_per_center = 100
grid_size = 3
scale = 0.2
n_clusters = grid_size * 3

def make_data(random_state, n_samples_per_center, grid_size, scale):
    random_state = check_random_state(random_state)
    centers = np.array([[i, j] for i in range(grid_size) 
                               for j in range(grid_size)])
    n_clusters_true, n_features = centers.shape

    noise = random_state.normal(
        scale=scale, size=(n_samples_per_center, centers.shape[1])
    )
    X = np.concatenate([c + noise for c in centers])
    y = np.concatenate([[i] * n_samples_per_center
                        for i in range(n_clusters_true)])
    return shuffle(X, y, random_state=random_state)

plt.figure()
plots = []
legends = []

cases = [
    (KMeans, "k-means++", {}),
    (KMeans, "random", {}),
    (MiniBatchKMeans, "k-means++", {'max_no_improvement': 3}),
    (MiniBatchKMeans, "random", {'max_no_improvement': 3, 'init_size': 500})
]

for factory, init, params in cases:
    print("Evaluation of %s with %s init" % (factory.__name__,init))
    inertia = np.empty((len(n_init_range), NUM_RUNS))

    for run_id in range(NUM_RUNS):
        X, y = make_data(run_id, n_samples_per_center,grid_size,scale)
        for i, n_init in enumerate(n_init_range):
            km = factory(n_clusters=n_clusters, init=init, random_state=run_id,n_init=n_init,**params).fit(X,y)
            inertia[i, run_id] = km.inertia_
    
    p = plt.errorbar(n_init_range, inertia.mean(axis=1),inertia.std(axis=1))
    plots.append(p[0])
    legends.append("%s with %s init" % (factory.__name__, init))

plt.xlabel('n_init')
plt.ylabel('inertia')
plt.legend(plots, legends)
plt.title('Mean inertia for various k-means init across %d runs' % NUM_RUNS)

X, y =make_data(random_state, n_samples_per_center, grid_size,scale)
km = MiniBatchKMeans(n_clusters=n_clusters,init='random',n_init=1, random_state=random_state).fit(X,y)

plt.figure()
for k in range(n_clusters):
    my_members = km.labels_ == k
    color = cm.nipy_spectral(float(k) / n_clusters, 1)
    plt.plot(X[my_members, 0],X[my_members, 1],'o',marker='.',c=color)
    cluster_center = km.cluster_centers_[k]
    plt.plot(cluster_center[0],cluster_center[1],'x',markerfacecolor=color,markeredgecolor='k',markersize=6)
    plt.title("Example cluster allocation with a single random init with MiniBatchKMeans")

plt.show()