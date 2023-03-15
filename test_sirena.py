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

# Loading hourly electricity data
infile  = sys.argv[1]
df = pd.read_csv(infile)
df = df.dropna() 

# Sorting by date and rearranging to obtain daily vectors of 24h 
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df.set_index("date",inplace=True)
df.sort_index(inplace=True)
df = pd.get_dummies(data=df, columns=['hour'])

for c, v in df.iteritems():
    if c != 'Value':
        df[c] = df[c]*df['Value']

df = df.drop(['Value'], axis=1)
df = df.groupby('date').sum()

y = df.index.dayofweek
X = df.to_numpy()

# SDOclust uses MAD to remove outliers cluster-wise 
# based on SDO outlierness scores
from pythresh.thresholds.mad import MAD
def remove_outs(y, s, c):
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
y_sdoc = remove_outs(y_sdoc, outscs_sdoc, 2)


# Visualizing clustering results with tSNE and PCA
from sklearn.manifold import TSNE
Xtsne = TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sca = StandardScaler()
sca.fit(X)
Xzs = sca.transform(X)
pca = PCA(n_components=2)
pca.fit(Xzs)
Xpca = pca.transform(Xzs)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
clu_hdbs, car_hdbs = np.unique(y_hdbs, return_counts=True)
clu_sdoc, car_sdoc = np.unique(y_sdoc, return_counts=True)
print("\nHDBSCAN (clusters):",clu_hdbs)
print("HDBSCAN (cardinality):", car_hdbs)
print("SDOclust (clusters):",clu_sdoc)
print("SDOclust (cardinality):", car_sdoc)

axes[0,0].scatter(Xtsne[:,0], Xtsne[:,1], s=50, c=y_hdbs, cmap='tab10', alpha=0.8, edgecolors='w')
axes[0,0].set_title("tSNE HDBSCAN")
axes[1,0].scatter(Xpca[:,0], Xpca[:,1], s=50, c=y_hdbs, cmap='tab10', alpha=0.8, edgecolors='w')
axes[1,0].set_title("PCA HDBSCAN")
axes[0,1].scatter(Xtsne[:,0], Xtsne[:,1], s=50, c=y_sdoc, cmap='tab10', alpha=0.8, edgecolors='w')
axes[0,1].set_title("tSNE SDOclust")
axes[1,1].scatter(Xpca[:,0], Xpca[:,1], s=50, c=y_sdoc, cmap='tab10', alpha=0.8, edgecolors='w')
axes[1,1].set_title("PCA SDOclust")
axes[0,2].scatter(Xtsne[:,0], Xtsne[:,1], s=50, c=y, cmap='tab10', alpha=0.8, edgecolors='w')
axes[0,2].set_title("tSNE Day-of-week")
axes[1,2].scatter(Xpca[:,0], Xpca[:,1], s=50, c=y, cmap='tab10', alpha=0.8, edgecolors='w')
axes[1,2].set_title("PCA Day-of-week")
ptname = "plots/sirena_2D-projections.pdf"
plt.savefig(ptname, dpi=200)
print("\nPlot", ptname, "has been created!")
plt.show()
plt.close()

# Estimating validity measurements and comparing HDBSCAN and SDOclust
from sklearn.metrics import silhouette_score
print("\nSil - HDBSCAN (inc. out)", silhouette_score(X, y_hdbs, metric='euclidean'))
print("Sil - SDOclust (inc. out)", silhouette_score(X, y_sdoc, metric='euclidean'))
print("Sil - HDBSCAN (no out)", silhouette_score(X[y_hdbs>-1], y_hdbs[y_hdbs>-1], metric='euclidean'))
print("Sil - SDOclust (no out)", silhouette_score(X[y_sdoc>-1], y_sdoc[y_sdoc>-1], metric='euclidean'))

from sklearn.metrics.cluster import adjusted_rand_score
print("\nSimilarity between HDBSCAN and SDOclust clustering")
print("ARI:", adjusted_rand_score(y_hdbs, y_sdoc))

print("\nOutliers found by HDBSCAN:", np.sum(y_hdbs==-1))
print("Outliers found by SDOclust:", np.sum(y_sdoc==-1))

# Plotting patterns and profiles found by HDBSCAN and SDOclust
cols = df.columns
idx = df.index
df2 = pd.DataFrame(X, columns = cols, index = idx)
df2['y_hdbs'] = y_hdbs
df2['y_sdoc'] = y_sdoc
df2['dow'] = y

algs = [y_hdbs,y_sdoc]
algs_name = ['y_hdbs','y_sdoc']
algorithms = ['HDBSCAN','SDOclust']

# Clusters are manually rellabeled for a better alignment and comparison in 
# the generated plots 
i_hdbs = [0,4,3,1,2,5,6,8,7]
i_sdoc = [0,3,7,5,2,4,6,1]
jas = [i_hdbs,i_sdoc]

maxcl = max(len(np.unique(y_hdbs)),len(np.unique(y_sdoc)))
fig, axes = plt.subplots(2, maxcl,figsize=(20, 5))
for i,a in enumerate(algs):
    clu, car = np.unique(a, return_counts=True)
    for j,c in enumerate(np.unique(a)):
        sja = jas[i]
        ja = sja[j]
        if np.sum(a==c)>0:
            df2c = df2[df2[algs_name[i]]==c].drop(['y_hdbs','y_sdoc'], axis=1).T   
            sns.lineplot(data=df2c, ax=axes[i,ja], legend=False, palette='Greys')
            axes[i,ja].set_ylim([0,80])
            axes[i,ja].set(xticklabels=[])  
            if i:
                axes[i,ja].set_xlabel("time (hours)")  
            if j==0:
                title = "outliers" + ' ('+str(car[j])+')'
                axes[i,ja].set_title(title)
                axes[i,ja].set_ylabel(algorithms[i]+"\nelec. (kWh)")  
            elif j<len(np.unique(a)):
                title = "P"+str(clu[ja]) + ' ('+str(car[j])+')'
                axes[i,ja].set_title(title)

    axes[0,maxcl-1].set_xlabel("time (hours)")  

fig.delaxes(axes[1,8])

ptname = "plots/sirena_patterns.pdf"
plt.savefig(ptname, dpi=200)
print("\nPlot", ptname, "has been created!")
plt.show()
plt.close()



