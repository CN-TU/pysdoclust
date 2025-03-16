from typing import Callable, Optional, Union
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, ClusterMixin

def check_adjacency_matrix(a: np.ndarray) -> bool:
    return (
        a.ndim == 2 and
        a.shape[0] == a.shape[1] and
        np.allclose(a, a.T) and  
        ((a == 0) | (a == 1)).all() and  
        not np.any(np.diag(a))  
    )

def distances_to_adjacency_matrix(distances: np.ndarray, threshold: float,) -> np.ndarray:
    N = distances.shape[0]
    adjacency_matrix = (distances < threshold[:, None]).astype(np.uint8)
    np.fill_diagonal(adjacency_matrix, 0) 
    return np.minimum(adjacency_matrix, adjacency_matrix.T)  

def _pairwise_distances(X: np.ndarray, metric="euclidean", n_jobs=None) -> np.ndarray:
    return pairwise_distances(X, metric=metric, n_jobs=n_jobs)

class ConnectedComponentsClustering(ClusterMixin, BaseEstimator):

    def __init__(
        self,
        zeta: float,
        chi: int,
        chi_min: int,
        chi_prop: float,
        metric: Union[str, Callable] = "euclidean",
        n_jobs: Optional[int] = None,
    ) -> None:
        self.threshold = 0
        self.zeta = zeta
        self.chi = chi
        self.chi_min = chi_min
        self.chi_prop = chi_prop
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray):
        X = self._validate_data(X, accept_sparse="csr")

        distances = _pairwise_distances(X, metric=self.metric, n_jobs=self.n_jobs)

        m, n = distances.shape
        chi = self.chi or max(int(m * self.chi_prop), self.chi_min)

        aux = np.partition(distances, chi, axis=1)[:, chi] 
        mean_aux = np.mean(aux)
        self.threshold = self.zeta * aux + (1 - self.zeta) * mean_aux 

        adjacency_matrix = distances_to_adjacency_matrix(distances, self.threshold)

        labels = connected_components(csgraph=csr_matrix(adjacency_matrix), directed=False, return_labels=True)[1]
        self.labels_ = labels

        return self

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.labels_

