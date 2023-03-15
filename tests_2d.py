
import sdo as sdo
from kmeansmm import KMeansMM
import hdbscan

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import sys
import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score

alg = sys.argv[1]

data_names = ["close", "s1", "separated", "iris", "mallcust","pima","r15","aggregation","skewed","asymmetric","high-noise", "low-noise", "rings","complex","moons"]

algorithms = ['hdbscan','kmeans--','sdoclust']

columns = ['dataset','samples', 'dimensions', 'outliers', 'clusters GT', 'clusters HDBSCAN', 'clusters kmeans--', 'clusters SDOclust', 'Sil HDBSCAN', 'Sil kmeans--', 'Sil SDOclust',  'Rand HDBSCAN', 'Rand kmeans--', 'Rand SDOclust']
df = pd.DataFrame(columns=columns)

fig = plt.figure(figsize=(20, 13))
gs = gridspec.GridSpec(3, 5) 

for d_ind, d_name in enumerate(data_names):

    file_name = "data2d/"+d_name+".csv"
    print("\n------- DATASET: ", d_name, "-------")
    dataset = np.genfromtxt(file_name, delimiter=',')

    data, y = dataset[:,0:2], dataset[:,2].astype(int)

    data = MinMaxScaler().fit_transform(data)
    [m,n] = data.shape

    GT_clusters = len(np.unique(y)) # num clusters in the GT
    outBool=False
    if min(y)==-1: # GT comes with outliers (label=-1)
        outBool=True
        GT_clusters = len(np.unique(y))-1

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

        contamination = np.sum(y==-1)
        outlier_labels = np.zeros(len(y))
        if contamination > 0:
            ind = np.argpartition(outlier_scores, -contamination)[-contamination:]
            outlier_labels[ind] = 1 

        if alg in algorithm:
            ax = plt.subplot(gs[d_ind])
            cm = plt.cm.get_cmap('Paired')
            sc = plt.scatter(data[outlier_labels==0,0], data[outlier_labels==0,1], c=crisp_labels[outlier_labels==0], cmap=cm, s=5)
            sc = plt.scatter(data[outlier_labels==1,0], data[outlier_labels==1,1], c='lightgray', s=5, alpha=0.5)
            plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
  
            ax.set_title(d_name)

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

    #out_perc = str(round(100*np.sum(y==-1)/len(data),1)) #+ '\%'
    new_row = [d_name, m, n, np.sum(y==-1), str(GT_clusters), numCh, numCk, numCs, Sh, Sk, Ss, ARh, ARk, ARs]
    df.loc[len(df.index)] = new_row

plt.tight_layout()
df = df.set_index("dataset")

import os
# Create folders for results 
os.makedirs('plots', exist_ok=True)
os.makedirs('tables', exist_ok=True)

# Saving tables with results and plots
ptname = "plots/2d-"+alg+".png"
ltxname = "tables/table2d.tex"
csvname = "tables/table2d.csv"
print("\nSaving results (csv): ", csvname)
df.to_csv(csvname)
print("Saving results (tex): ", ltxname)
df.style.to_latex(ltxname, hrules=True)
print("Saving plots: ", ptname)
plt.savefig(ptname)

