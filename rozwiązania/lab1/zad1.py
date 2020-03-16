import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds

X, y = ds.make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,
                              n_clusters_per_class=2, weights=None, flip_y=0.05, random_state=1)

pt = plt.scatter(X[:,0], X[:,1], c=y)
plt.savefig("plot1")

result = np.concatenate((X, y[:, np.newaxis]), axis=1)
np.savetxt("problem1.csv", result, delimiter='')