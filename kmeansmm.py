import numpy as np
import scipy.spatial.distance as distance

class KMeansMM:
    def __init__(self, n_clusters=10, l=5, max_iter=1000, tol=0.0001):
        self.k = n_clusters
        self.l = l
        self.centroids = None
        self.max_iter = max_iter
        self.tol = tol
            
    def fit(self, X):
        perm = np.random.permutation(X.shape[0])
        idx = perm[:self.k]
        centroids = X[idx,:]
        previous_state = centroids.copy()
        
        for i in range(self.max_iter):
            distances = distance.cdist(X, centroids)
            labels = np.argmin(distances, axis=1)
            mindists = np.min(distances, axis=1)
            
            L = (np.argsort(mindists)[::-1])[:self.l]
            labels[L] = -1 #remove outliers from the assigned cluster (assign special cluster/label -1)
            
            for cl in range(self.k):
                centroids[cl] = np.mean(X[labels == cl,:], axis=0)
            
            centroids_shift = np.sum(np.sqrt(np.sum(((previous_state - centroids) ** 2), axis=1)))
            if centroids_shift < self.tol:
                break
        
        self.centroids = centroids
        
        return self

    def predict(self, X):
        distances = distance.cdist(X,self.centroids)
        labels = np.argmin(distances, axis=1)
        mindists = np.min(distances, axis=1)

        L = (np.argsort(mindists)[::-1])[:self.l]
        labels[L] = -1 #remove outliers from the assigned cluster
        return labels, mindists
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
