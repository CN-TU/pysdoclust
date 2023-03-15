
import sdo as sdo
from kmeansmm import KMeansMM
import hdbscan

import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score

algorithms = ['hdbscan','kmeans--','sdoclust']

columns = ['dataset','samples', 'dimensions', 'outliers', 'clusters GT', 'clusters HDBSCAN', 'clusters kmeans--', 'clusters SDOclust', 'Sil HDBSCAN', 'Sil kmeans--', 'Sil SDOclust',  'Rand HDBSCAN', 'Rand kmeans--', 'Rand SDOclust']
df = pd.DataFrame(columns=columns)

# datasets
data_names = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15',
    'c16','c17','c18','c19','c20','d3','d4','d5','d6','d7','d8','d9','d10','d11','d12','d13','d14','d15',
    'd32','d64','dc','dq','dk','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15',
    'f16','f17','f18','f19','f20','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15',
    'h16','h17','h18','h19','h20','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','n11','n12','n13','n14','n15',
    'n16','n17','n18','n19','n20','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15',
    'p16','p17','p18','p19','p20','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15',
    'x16','x17','x18','x19','x20']


for d_ind, d_name in enumerate(data_names):

    file_name = "dataMd/"+d_name
    dataset = np.genfromtxt(file_name, delimiter=',')

    print("\n------- DATASET: ", d_name, "-------")
    data, y = dataset[:,0:-1], dataset[:,-1].astype(int)
    [m,n] = data.shape

    GT_clusters = len(np.unique(y)) # num clusters in the GT
    outBool=False
    if min(y)==-1: # GT comes with outliers (label=-1)
        outBool=True
        GT_clusters = len(np.unique(y))-1

    # normalize dataset
    data = MinMaxScaler().fit_transform(data)

    for algorithm in algorithms:

        if algorithm == 'kmeans--':
            crisp_labels, outlier_scores = KMeansMM(n_clusters=GT_clusters, l=np.sum(y==-1)).fit_predict(data)
        elif algorithm == 'hdbscan':
            hdbs = hdbscan.HDBSCAN()
            crisp_labels = hdbs.fit_predict(data)
            outlier_scores = hdbs.outlier_scores_
        elif algorithm == 'sdoclust':
            #crisp_labels, memberships, outlier_scores, observers, obs_labels = sdo.SDOclust(data)
            sdoclust = sdo.SDOclust()
            sdoclust = sdoclust.fit(data)
            crisp_labels = sdoclust.predict(data)
            outlier_scores = sdoclust.outlierness(data)

        # using contamination to match the expected number of outliers according to the GT
        contamination = np.sum(y==-1)
        outlier_labels = np.zeros(len(y))
        if contamination > 0:
            ind = np.argpartition(outlier_scores, -contamination)[-contamination:]
            outlier_labels[ind] = 1 
            crisp_labels[ind] = -1

        # Silhouette makes sense after outlier removal
        try:
            S = silhouette_score(data[outlier_labels==0,:], crisp_labels[outlier_labels==0], metric='euclidean')
            #S = silhouette_score(data, crisp_labels, metric='euclidean')
        except:
            S = np.nan
        AR = adjusted_rand_score(y, crisp_labels)

        num_clusters = len(np.unique(crisp_labels))
        if np.min(crisp_labels)==-1:
            num_clusters = num_clusters-1

        print('Algorithm:', algorithm)
        print('- clusters(GT):', str(GT_clusters), ', clusters(pred):', str(num_clusters) )
        print('- Sil:', round(S,2), ', Rand:', round(AR,2))
        
        clust_diff = '='
        if GT_clusters > num_clusters:
            clust_diff = '-'+ str(GT_clusters-num_clusters)
        elif GT_clusters < num_clusters:
            clust_diff = '+'+ str(num_clusters-GT_clusters)

        if algorithm == 'kmeans--':
            Sk, ARk, numCk = str(round(S,2)), str(round(AR,2)), clust_diff
        elif algorithm == 'hdbscan':
            Sh, ARh, numCh = str(round(S,2)), str(round(AR,2)), clust_diff
        elif algorithm == 'sdoclust':
            Ss, ARs, numCs = str(round(S,2)), str(round(AR,2)), clust_diff

    new_row = [d_name, m, n, np.sum(y==-1), str(GT_clusters), numCh, numCk, numCs, Sh, Sk, Ss, ARh, ARk, ARs]
    df.loc[len(df.index)] = new_row

df = df.set_index("dataset")

import os
# Create folder for results 
os.makedirs('tables', exist_ok=True)

# Saving tables with results
ltxname = "tables/tableMd.tex"
csvname = "tables/tableMd.csv"
print("\nSaving results (csv): ", csvname)
df.to_csv(csvname)
print("Saving results (tex): ", ltxname)
df.style.to_latex(ltxname, hrules=True)

