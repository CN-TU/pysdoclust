"""
SDO (Sparse Data Observers)

Python library for the SDO (outlier scoring) and SDOclust (clustering) algorithms.
"""

__version__ = "0.5"
__author__ = 'Félix Iglesias Vázquez'
__credits__ = 'TU Wien, Inst. of Telecomm., CN Group'


import math
import numpy as np
import scipy.spatial.distance as distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sdoclust.clust import ConnectedComponentsClustering

def extend_labels(X, O, l, knn=5, method="brute", index=None, chunksize=None):
    m, n = X.shape
    cl = len(np.unique(l))

    if chunksize is None:
        chunksize = m

    C = np.zeros((m, cl))

    if method == "brute":
        for i in range(0, m, chunksize):
            dist = distance.cdist(X[i:i+chunksize], O)
            closest = np.argpartition(dist, knn, axis=1)[:, :knn]
            lknn = l[closest]
            for j in range(cl):
                C[i:i+chunksize, j] = np.sum(lknn == j, axis=1)

    elif method == "faiss":
        if index is None:
            raise ValueError("FAISS index must be provided.")
        for i in range(0, m, chunksize):
            _, closest = index.search(X[i:i+chunksize].astype(np.float32), knn)
            lknn = l[closest]
            for j in range(cl):
                C[i:i+chunksize, j] = np.sum(lknn == j, axis=1)

    elif method == "pynndescent":
        if index is None:
            raise ValueError("PyNNDescent index must be provided.")
        closest, _ = index.query(X, k=knn)
        for j in range(cl):
            C[:, j] = np.sum(l[closest] == j, axis=1)

    else:
        raise ValueError(f"Unknown method '{method}'. Options available: 'brute', 'faiss', or 'pynndescent'.")

    return C / knn, np.argmax(C, axis=1)    


def graph_clust(x, zeta, chi, chi_min, chi_prop):
    model = ConnectedComponentsClustering(zeta, chi, chi_min, chi_prop, metric="euclidean", n_jobs=-1)
    l = model.fit_predict(x)
    return l

def hbdiscret(O):
    [m,n] = O.shape
    Od = np.copy(O)
    nbins = int(20*np.sqrt(m))
    for i in range(n):
        Oi = O[:,i]
        bins = np.histogram_bin_edges(Oi, bins=nbins)
        midbins = bins[:-1]+(bins[1:]-bins[:-1])/2
        midbins = np.append(midbins, np.max(Oi))
        ind = np.digitize(Oi, bins)-1
        Od[:,i] = midbins[ind]
    Od = np.unique(Od, axis=0)
    kd = Od.shape[0]
    return Od, kd

def sample_size (N, s, e):
	z=1.96
	num = N * pow(z,2) * pow(s,2)
	den = (N-1) * pow(e,2) + pow(z,2) * pow(s,2)
	n = int(math.floor(num/den))
	return n

def smoothing (O, x, f):
    [k,n] = O.shape
    dist = distance.cdist(O, O)
    dist_sorted = np.sort(dist, axis=1)
    O_sorted = np.argsort(dist, axis=1)
    closest = O_sorted[:,1:x+1]
    dist_closest = dist_sorted[:,1:x+1]
    dmean = np.mean(dist_closest, axis=1)
    inc = (dist_closest - dmean[:, None])*f*-1
    Ohat = (O / np.linalg.norm(O, axis=1)[:, None])

    for i in range(k):
        O[i,:] =  O[i,:] + np.sum(Ohat[closest[i]]*inc[i][:,None], axis=0)
    return O
	

class SDO:
    def __init__(self, x=5, qv=0.3, hbs=False, smooth=False, smooth_f=0.25, rseed=0, k=None, q=None, chunksize=None, method="brute"):
        self.x = x
        self.qv = qv
        self.hbs = hbs
        self.smooth = smooth
        self.smooth_f = smooth_f
        self.rseed = rseed
        self.k = k
        self.q = q
        self.chunksize = chunksize
        self.method = method  # "brute", "faiss", "pynndescent"
        self.O = None
        self.index = None  # For the FAISS or pyNNDescent indexing
        self._faiss = None
        self._pynndescent = None
        if method == "faiss":
            import faiss
            self._faiss = faiss
        elif method == "pynndescent":
            import pynndescent
            self._pynndescent = pynndescent

    def fit(self, X):
        m, n = X.shape
        
        if self.k is None:
            Xt = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2)
            Xp = pca.fit_transform(Xt)
            sigma = max(np.std(Xp), 1)
            error = 0.1 * np.std(Xp)
            self.k = sample_size(m, sigma, error)
            k = self.k
        else:
            k = self.k

        chunksize = self.chunksize or m
        np.random.seed(self.rseed)
        index = np.random.permutation(m)
        O = X[index[:k]]
        
        if self.hbs:
            O, k = hbdiscret(O)
        if self.smooth:
            O = smoothing(O, self.x, self.smooth_f)
        
        P = np.zeros(k)
        
        if self.method == "faiss":
            self.index = self._faiss.IndexFlatL2(O.shape[1])
            self.index.add(O.astype(np.float32))
        elif self.method == "pynndescent":
            self.index = self._pynndescent.NNDescent(O.astype(np.float32), n_neighbors=self.x, metric="euclidean")
        
        for i in range(0, m, chunksize):
            X_chunk = X[i:(i + chunksize)]
            
            if self.method == "brute":
                dist = distance.cdist(X_chunk, O)
                closest = np.argsort(dist, axis=1)[:, :self.x].flatten()
            elif self.method == "faiss":
                _, closest = self.index.search(X_chunk.astype(np.float32), self.x)
                closest = closest.flatten()
            elif self.method == "pynndescent":
                closest, _ = self.index.query(X_chunk, k=self.x)
                closest = closest.flatten()
            
            P += np.count_nonzero(closest[:, np.newaxis] == np.arange(k), 0)
        
        q = np.quantile(P, self.qv) if self.q is None else self.q
        self.O = O[P >= q]
        
        if self.method == "faiss":
            self.index = self._faiss.IndexFlatL2(self.O.shape[1])
            self.index.add(self.O.astype(np.float32))
        elif self.method == "pynndescent":
            self.index = self._pynndescent.NNDescent(self.O.astype(np.float32), n_neighbors=self.x, metric="euclidean")
            
        return self

    def get_observers(self):
        return self.O, self.index

    def set_observers(self, O):
        self.O = O
        return self
        
    def predict(self, X):
        m, n = X.shape
        chunksize = self.chunksize or m
        O = self.O
        y = np.zeros(m)
        
        for i in range(0, m, chunksize):
            X_chunk = X[i:(i + chunksize)]
            
            if self.method == "brute":
                dist = distance.cdist(X_chunk, O)
            elif self.method == "faiss":
                dist, _ = self.index.search(X_chunk.astype(np.float32), self.x)
            elif self.method == "pynndescent":
                _, dist = self.index.query(X_chunk, k=self.x)
            
            y[i:(i + chunksize)] = np.median(np.sort(dist, axis=1)[:, :self.x], axis=1)
        
        return y

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
        

class SDOclust:
    def __init__(self, x=5, qv=0.3, hbs=False, smooth=False, smooth_f=0.25, rseed=5, zeta=0.6, 
                 chi_min=8, chi_prop=0.05, e=3, chi=None, cb_outlierness=False, xc=None, 
                 k=None, q=None, chunksize=None, method="brute"):
        self.x = x
        self.qv = qv
        self.hbs = hbs
        self.smooth = smooth
        self.smooth_f = smooth_f
        self.rseed = rseed
        self.zeta = zeta
        self.chi_min = chi_min
        self.chi_prop = chi_prop
        self.e = e
        self.chi = chi
        self.cb_outlierness = cb_outlierness
        self.xc = xc if xc is not None else x
        self.k = k
        self.q = q
        self.chunksize = chunksize
        self.method = method  
        self._faiss = None
        self._pynndescent = None
        self.index = None
        self.O = None
        self.ol = None
        self.kmp = None
        self.sdo = None

    def fit(self, X):
        self.sdo = SDO(self.x, self.qv, self.hbs, self.smooth, self.smooth_f, 
                       self.rseed, self.k, self.q, self.chunksize, method=self.method)  
        self._faiss = self.sdo._faiss 
        self._pynndescent = self.sdo._pynndescent
        
        self.sdo.fit(X)
        self.O, self.index = self.sdo.get_observers()

        self.ol = graph_clust(self.O, self.zeta, self.chi, self.chi_min, self.chi_prop)

        # Remove clusters with too few members
        ind, count = np.unique(self.ol, return_counts=True)
        toremove = np.zeros(len(self.O), dtype=bool)
        for i in ind:
            if count[i] <= self.e:
                toremove[self.ol == i] = True

        if np.all(toremove):  # Prevent removing all observers
            toremove[:] = False

        self.O = self.O[~toremove]
        self.ol = self.ol[~toremove]
        
        if self.method == "faiss":
            self.index = self._faiss.IndexFlatL2(self.O.shape[1])
            self.index.add(self.O.astype(np.float32))
        elif self.method == "pynndescent":
            self.index = self._pynndescent.NNDescent(self.O.astype(np.float32), n_neighbors=self.x, metric="euclidean")

        # Relabel clusters to be consecutive
        unique_labels = np.unique(self.ol)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self.ol = np.array([label_map[label] for label in self.ol])

        self.kmp = self.O.shape[0] / X.shape[0]

        return self

    def predict(self, X, return_membership=False, xc=None):
        xc = xc if xc is not None else self.xc
        m, c = extend_labels(X, self.O, self.ol, xc, self.method, self.index, self.chunksize)
        return (c, m) if return_membership else c

    def fit_predict(self, X, return_membership=False, xc=None):
        self.fit(X)
        return self.predict(X, return_membership=return_membership, xc=xc)

    def get_observers(self):
        return self.O, self.index

    def outlierness(self, X, x=None):
        x = x if x is not None else self.x
        return self.sdo.predict(X, x)

    def update(self, X):
        k = self.O.shape[0]
        m, _ = X.shape
        k_new_obs = max(1, int(m * self.kmp))
        k_excess = k + k_new_obs

        index = np.random.permutation(m)
        newO = X[index[:k_new_obs]]
        O = np.vstack((self.O, newO))

        P = np.zeros(k_excess)

        chunksize = self.chunksize or m
        for i in range(0, m, chunksize):
            dist = distance.cdist(X[i:i+chunksize], O)
            dist_sorted = np.argsort(dist, axis=1)
            closest = dist_sorted[:, :self.x].flatten()
            P += np.count_nonzero(closest[:, np.newaxis] == np.arange(k_excess), axis=0)

        # Remove least-represented observers
        toremove = np.argsort(P)[:k_new_obs]
        O = np.delete(O, toremove, axis=0)

        self.O = O
        self.ol = graph_clust(O, self.zeta, self.chi, self.chi_min, self.chi_prop)

        return self

    def update_predict(self, X, return_membership=False, xc=None):
        self.update(X)
        return self.predict(X, return_membership=return_membership, xc=xc)

