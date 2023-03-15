import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

import warnings
warnings.filterwarnings("ignore")

import os
# Create folders for results 
os.makedirs('plots', exist_ok=True)

# Loading preprocessed MAWI network traffic data
infile  = sys.argv[1]
df = pd.read_csv(infile)
X = df.to_numpy()[:,:-1]
y = df['proto'].to_numpy()


# Normalizing data based on quantiles
from sklearn.preprocessing import quantile_transform
X = quantile_transform(X, n_quantiles=200, random_state=0, copy=True)

# Plotting histograms of features
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
sns.histplot(data=df, x="flowDurationMilliseconds", hue='proto', palette='tab10', ax=axes[0,0])
sns.histplot(data=df, x="pkts_forward", hue='proto', palette='tab10', ax=axes[0,1])
sns.histplot(data=df, x="mode_ipLen_forward", hue='proto', palette='tab10', ax=axes[0,2])
sns.histplot(data=df, x="IAT_mean_forward", hue='proto', palette='tab10', ax=axes[0,3])
sns.histplot(data=df, x="pkts_backward", hue='proto', palette='tab10', ax=axes[1,1])
sns.histplot(data=df, x="mode_ipLen_backward", hue='proto', palette='tab10', ax=axes[1,2])
sns.histplot(data=df, x="IAT_mean_backward", hue='proto', palette='tab10', ax=axes[1,3])
sns.histplot(data=df, x="proto", palette='tab10', hue='proto', ax=axes[1,0])
#fig.delaxes(axes[1,0])
fig.suptitle("Histograms of the network traffic data sample", fontsize=14)

ptname = "plots/mawi_sample_histograms.pdf"
plt.savefig(ptname, dpi=200)
print("\nPlot", ptname, "has been created!")
plt.show()
plt.close()

# Array for the color indexing of protocols
data = df.to_numpy()
X = data[:,:-1]
y = data[:,-1]
ycol = y.astype(int)
ycol[y==6],ycol[y==17] = 2,3

# SDOclust uses MAD to remove outliers cluster-wise 
# based on SDO outlierness scores
from pythresh.thresholds.mad import MAD
def remove_outs(y, s):
    s = (s-min(s))/(max(s)-min(s))
    for i in np.unique(y):
        si = s[y==i]
        thres = MAD()
        thres.eval(si)
        y[(y == i) & (s>=thres.thresh_) ] = -1
    return y

# Clustering with SDOclust and HDBSCAN
import sdo as sdo
import hdbscan

hdbs = hdbscan.HDBSCAN()
y_hdbs = hdbs.fit_predict(X)

sdoclust = sdo.SDOclust()
sdoclust = sdoclust.fit(X)
y_sdoc = sdoclust.predict(X)
outscs_sdoc = sdoclust.outlierness(X)
y_sdoc = remove_outs(y_sdoc, outscs_sdoc)

clu_hdbs, car_hdbs = np.unique(y_hdbs, return_counts=True)
clu_sdoc, car_sdoc = np.unique(y_sdoc, return_counts=True)
print("\nHDBSCAN (clusters):",clu_hdbs)
print("HDBSCAN (cardinality):", car_hdbs)
print("SDOclust (clusters):",clu_sdoc)
print("SDOclust (cardinality):", car_sdoc)

# Visualizing clustering results with tSNE and PCA
print("\nPerforming tSNE projection...")
from sklearn.manifold import TSNE
Xtsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)

print("\nPerforming PCA projection...")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sca = StandardScaler()
sca.fit(X)
Xzs = sca.transform(X)
pca = PCA(n_components=2)
pca.fit(Xzs)
Xpca = pca.transform(Xzs)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
print(np.unique(y_hdbs, return_counts=True))
print(np.unique(y_sdoc, return_counts=True))
cmap = plt.get_cmap('Accent', len(np.unique(ycol)))
axes[0].scatter(Xtsne[:,0], Xtsne[:,1], s=50, c=ycol, cmap=cmap, alpha=0.8, edgecolors='w')
axes[0].set(yticklabels=[]),axes[0].set(xticklabels=[])
axes[0].set_title("IP Protocol")
axes[0].set_xlabel("tSNE-1"),axes[0].set_ylabel("tSNE-2")
axes[0].text(1.14, 0.11, "others", fontsize=12, transform=axes[0].transAxes)
axes[0].text(1.14, 0.34, "ICMP", fontsize=12, transform=axes[0].transAxes)
axes[0].text(1.14, 0.60, "TCP", fontsize=12, transform=axes[0].transAxes)
axes[0].text(1.14, 0.85, "UDP", fontsize=12, transform=axes[0].transAxes)
mappable = axes[0].collections[0]
fig.colorbar(mappable=mappable, ax=axes[0], ticks=[])
cmap = plt.get_cmap('nipy_spectral', len(np.unique(y_hdbs)))
axes[1].scatter(Xtsne[:,0], Xtsne[:,1], s=50, c=y_hdbs, cmap=cmap, alpha=0.8, edgecolors='w')
axes[1].set(yticklabels=[]),axes[1].set(xticklabels=[])
axes[1].set_title("HDBSCAN") 
axes[1].set_xlabel("tSNE-1"),axes[1].set_ylabel("tSNE-2")
axes[1].text(1.15, 0.36, "clusters (117)", fontsize=12, rotation='vertical', transform=axes[1].transAxes)
axes[1].text(1.15, 0.00, "outs.", fontsize=12, transform=axes[1].transAxes)
mappable = axes[1].collections[0]
fig.colorbar(mappable=mappable, ax=axes[1], ticks=[])
cmap = plt.get_cmap('nipy_spectral', len(np.unique(y_sdoc)))
axes[2].scatter(Xtsne[:,0], Xtsne[:,1], s=50, c=y_sdoc, cmap=cmap, alpha=0.8, edgecolors='w')
axes[2].set(yticklabels=[]),axes[2].set(xticklabels=[]) 
axes[2].set_title("SDOclust") 
axes[2].set_xlabel("tSNE-1"),axes[2].set_ylabel("tSNE-2")
axes[2].text(1.15, 0.39, "clusters (8)", fontsize=12, rotation='vertical', transform=axes[2].transAxes)
axes[2].text(1.15, 0.00, "outs.", fontsize=12, transform=axes[2].transAxes)
mappable = axes[2].collections[0]
fig.colorbar(mappable=mappable, ax=axes[2], ticks=[])

ptname = "plots/mawi_2D-projections.pdf"
plt.savefig(ptname, dpi=200)
print("\nPlot", ptname, "has been created!")
plt.show()
plt.close()

from sklearn.metrics import silhouette_score
print("\nSil - HDBSCAN (inc. out)", silhouette_score(X, y_hdbs, metric='euclidean'))
print("Sil - SDOclust (inc. out)", silhouette_score(X, y_sdoc, metric='euclidean'))
print("Sil - HDBSCAN (no out)", silhouette_score(X[y_hdbs>-1], y_hdbs[y_hdbs>-1], metric='euclidean'))
print("Sil - SDOclust (no out)", silhouette_score(X[y_sdoc>-1], y_sdoc[y_sdoc>-1], metric='euclidean'))

from sklearn.metrics.cluster import adjusted_rand_score
print("\nARI - HDBSCAN vs Protocols: ", adjusted_rand_score(y, y_hdbs))
print("ARI - SDOclust vs Protocols: ",adjusted_rand_score(y, y_sdoc))

print("\nOutliers found by HDBSCAN:", np.sum(y_hdbs==-1))
print("Outliers found by SDOclust:", np.sum(y_sdoc==-1))


