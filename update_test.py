
import sdo as sdo
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score

import matplotlib.pyplot as plt
from matplotlib import gridspec

cm = plt.cm.get_cmap('Paired')
data_names = ["rings"]

for d_ind, d_name in enumerate(data_names):

    file_name = "data2d/"+d_name+".csv"
    dataset = np.genfromtxt(file_name, delimiter=',')

    print("\n------- DATASET: ", d_name, "-------")
    data, y = dataset[:,0:-1], dataset[:,-1].astype(int)

    # normalize dataset
    data = MinMaxScaler().fit_transform(data)
    [m,n] = data.shape

    ci = 20
    chunksize = 100
    reps = int(m/chunksize)
    first_time = 1

    sdoclust = sdo.SDOclust()

    for i in range(ci-1,reps): 

        if first_time:
            first_time = 0
            datachunk = data[:ci*chunksize,:]
            ychunk = y[:ci*chunksize]
            #sdoclust = sdoclust.fit(datachunk)
            #crisp_labels = sdoclust.predict(datachunk)
            crisp_labels = sdoclust.fit_predict(datachunk)
            outlier_scores = sdoclust.outlierness(datachunk)
            observers = sdoclust.get_observers()
        
        else:      
            datachunk = data[i*chunksize:(i+1)*chunksize,:]
            ychunk = y[i*chunksize:(i+1)*chunksize]
            #sdoclust = sdoclust.update(datachunk)
            #crisp_labels = sdoclust.predict(datachunk)
            crisp_labels = sdoclust.update_predict(datachunk)
            outlier_scores = sdoclust.outlierness(datachunk)
            observers = sdoclust.get_observers()

        print("chunk:", i, "shape:", datachunk.shape)
        contamination = np.sum(ychunk==-1)
        outlier_labels = np.zeros(len(ychunk))
        if contamination > 0:
            ind = np.argpartition(outlier_scores, -contamination)[-contamination:]
            outlier_labels[ind] = 1 

        GT_clusters = len(np.unique(ychunk)) # num clusters in the GT
        outBool=False
        if min(ychunk)==-1: # GT comes with outliers (label=-1)
            outBool=True
            GT_clusters = len(np.unique(ychunk))-1


        # Silhouette makes sense after outlier removal
        try:
            S = silhouette_score(datachunk[outlier_labels==0,:], crisp_labels[outlier_labels==0], metric='euclidean')
            #S = silhouette_score(data, crisp_labels, metric='euclidean')
        except:
            S = np.nan
        AR = adjusted_rand_score(ychunk, crisp_labels)

        num_clusters = len(np.unique(crisp_labels))
        if np.min(crisp_labels)==-1:
            num_clusters = num_clusters-1

        print('clusters(GT):', str(GT_clusters), ', clusters(pred):', str(num_clusters), ', Sil:', round(S,2), ', Rand:', round(AR,2), ', k:', observers.shape[0])

        plt.scatter(datachunk[outlier_labels==0,0], datachunk[outlier_labels==0,1], c=crisp_labels[outlier_labels==0], cmap=cm, s=5)
        plt.scatter(datachunk[outlier_labels==1,0], datachunk[outlier_labels==1,1], c='lightgray', s=5, alpha=0.5)
        plt.show()


