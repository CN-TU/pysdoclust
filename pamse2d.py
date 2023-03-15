
import sdo as sdo
import numpy as np
import pandas as pd
import sys

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score

param = sys.argv[1]

data_names = ["close", "s1", "separated", "iris", "mallcust","pima","r15","aggregation","skewed","asymmetric","high-noise", "low-noise", "rings","complex","moons"]

columns = ['dataset','nGT==nSDOc', 'Sil','AR','pamval']
df = pd.DataFrame(columns=columns)

if param == 'zeta':
    liml, limh = 0, 1 
    reps = 12
    pamval = np.linspace(liml, limh, reps)

elif param == 'chi':
    liml, limh = 8, 20 
    pamval = np.arange(liml, limh, 1, dtype=int)
    reps = len(pamval)

elif param == 'chi_min':
    liml, limh = 5, 15 
    pamval = np.arange(liml, limh, 1, dtype=int)
    reps = len(pamval)

elif param == 'chi_prop':
    liml, limh = 0.01, 0.12 
    pamval = np.arange(liml, limh, 0.01, dtype=float)
    reps = len(pamval)

elif param == 'e':
    liml, limh = 1, 5 
    pamval = np.arange(liml, limh, 1, dtype=int)
    reps = len(pamval)

elif param == 'smooth_f':
    liml, limh = 0, 1 
    pamval = np.arange(liml, limh, 0.1, dtype=float)
    reps = len(pamval)

elif param == 'hbs':
    pamval = [True, False]
    reps = len(pamval)

randreps = 10


for d_ind, d_name in enumerate(data_names):

    file_name = "data2d/"+d_name+".csv"
    print("\n------- DATASET: ", d_name, "-------")
    dataset = np.genfromtxt(file_name, delimiter=',')

    data, y = dataset[:,0:2], dataset[:,2].astype(int)

    # normalize dataset
    data = MinMaxScaler().fit_transform(data)
    [m,n] = data.shape

    GT_clusters = len(np.unique(y)) # num clusters in the GT
    outBool=False
    if min(y)==-1: # GT comes with outliers (label=-1)
        outBool=True
        GT_clusters = len(np.unique(y))-1


    for i in range(randreps):

        for j in range(reps):

            if param == 'zeta':
                sdoclust = sdo.SDOclust(zeta=pamval[j], rseed=i)
            elif param == 'chi':
                sdoclust = sdo.SDOclust(chi=pamval[j], rseed=i)
            elif param == 'chi_min':
                sdoclust = sdo.SDOclust(chi_min=pamval[j], rseed=i)
            elif param == 'chi_prop':
                sdoclust = sdo.SDOclust(chi_prop=pamval[j], rseed=i)
            elif param == 'e':
                sdoclust = sdo.SDOclust(e=pamval[j], rseed=i)
            elif param == 'smooth_f':
                sdoclust = sdo.SDOclust(smooth=True, smooth_f=pamval[j], rseed=i)
            elif param == 'hbs':
                sdoclust = sdo.SDOclust(hbs=pamval[j], rseed=i)

            sdoclust = sdoclust.fit(data)
            crisp_labels = sdoclust.predict(data)
            outlier_scores = sdoclust.outlierness(data)

            contamination = np.sum(y==-1)
            outlier_labels = np.zeros(len(y))
            if contamination > 0:
                ind = np.argpartition(outlier_scores, -contamination)[-contamination:]
                outlier_labels[ind] = 1 

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

            print(j, ': clusters(GT):', str(GT_clusters), ', clusters(pred):', str(num_clusters), ', Sil:', round(S,2), ', Rand:', round(AR,2), ', PamVal:', pamval[j])

            matched = 0
            if GT_clusters == num_clusters:
                matched = 1
            new_row = [d_name, matched, S, AR, pamval[j]]
            df.loc[len(df.index)] = new_row

for j in range(reps):
    rdf = df[df['pamval'] == pamval[j]]
    print('pamval:', pamval[j], ', Sil mean: ', np.mean(rdf['Sil'].to_numpy()))

for j in range(reps):
    rdf = df[df['pamval'] == pamval[j]]
    print('pamval:', pamval[j], ', AR mean: ', np.mean(rdf['AR'].to_numpy()))

#csvname = param+"2d.csv"
#df.to_csv(csvname)

