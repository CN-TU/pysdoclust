import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cddiagram

import sys

f2d  = sys.argv[1]
fMd  = sys.argv[2]

df = pd.read_csv(f2d)
dfMd = pd.read_csv(fMd)

df = pd.concat([df, dfMd])
df = df.dropna() 

# Sil comparison
dh = df[['dataset','Sil HDBSCAN']].copy()
dh['alg'] = 'HDBSCAN'
dh = dh[['alg'] + [x for x in dh.columns if x != 'alg']]
dk = df[['dataset','Sil kmeans--']].copy()
dk['alg'] = 'k-means--'
dk = dk[['alg'] + [x for x in dk.columns if x != 'alg']]
ds = df[['dataset','Sil SDOclust']].copy()
ds['alg'] = 'SDOclust'
ds = ds[['alg'] + [x for x in ds.columns if x != 'alg']]
dhks = np.vstack((dh.to_numpy(),dk.to_numpy(),ds.to_numpy()))

dfdiagS = pd.DataFrame(data=dhks, index=None, columns=['classifier_name','dataset_name','accuracy'], dtype=None, copy=False)

titleS = 'CD graph for Silhouette scores'
outputFS = 'cd_graph_sil'
cddiagram.draw_cd_diagram(df_perf=dfdiagS, title=titleS, labels=True, filename=outputFS)


# Rand comparison
dh = df[['dataset','Rand HDBSCAN']].copy()
dh['alg'] = 'HDBSCAN'
dh = dh[['alg'] + [x for x in dh.columns if x != 'alg']]
dk = df[['dataset','Rand kmeans--']].copy()
dk['alg'] = 'k-means--'
dk = dk[['alg'] + [x for x in dk.columns if x != 'alg']]
ds = df[['dataset','Rand SDOclust']].copy()
ds['alg'] = 'SDOclust'
ds = ds[['alg'] + [x for x in ds.columns if x != 'alg']]
dhks = np.vstack((dh.to_numpy(),dk.to_numpy(),ds.to_numpy()))

dfdiagR = pd.DataFrame(data=dhks, index=None, columns=['classifier_name','dataset_name','accuracy'], dtype=None, copy=False)

titleR = 'CD graph for ARI scores'
outputFR = 'cd_graph_ARI'
cddiagram.draw_cd_diagram(df_perf=dfdiagR, title=titleR, labels=True, filename=outputFR)

print("\n"+titleS, "saved in:", outputFS+'.csv' )
print(titleR, "saved in:", outputFR+'.csv' )
