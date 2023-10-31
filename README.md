
# SDOclust Evaluation Tests

Tests and Experiments conducted for the paper:
**SDOclust: Clustering with Sparse Data Observers**

Please, if you use SDOclust, refer to the original work as:

        F. Iglesias, T. Zseby, A. Hartl, A. Zimek. *SDOclust: Clustering Based on Sparse Data Observers*. 16th International Conference on Similarity Search and Applications, SISAP 2023. Oct 9-11, A Coruña, Spain. To be published...

(SDOclust was awarded with the "Best Student Paper Finalist" in [SISAP 2023](https://www.sisap.org/2023/) !!)

Note that [tables], [plots] and [cddiag] folders contain results as provided in the paper. By running the scripts below, files with results will be overwritten.

## Main python files
- *dependencies.py*: installs required python packages
- *tests_2d.py*: runs 2d experiments
- *tests_Md.py*: runs multi-dimensional experiments
- *test_mawi.py*: runs experiments with real network traffic data from MAWI captures
- *test_sirena.py*: runs experiments with real electricity consumption data from the Sirena project
- *sdo.py*: sdoclust functions
- *pamse2d.py*: script for sensitivity analysis on parameters
- *update_test.py*: script to show SDOclust in update modus
- *gbc.py*: graph-based clustering implementation (based on [https://github.com/dayyass/graph-based-clustering](https://github.com/dayyass/graph-based-clustering))
- *kmeansmm.py*: k-means-- implementation (based on [https://github.com/Strizzo/kmeans--](https://github.com/Strizzo/kmeans--))

## Install dependencies

        $ python3 dependencies.py 

## 2D experiments

        $ python3 tests_2d.py <option>

options: 'sdoclust', 'hdbscan', 'kmeans--'. The option is only to select the algorithm for plotting figures within the [plots] folder.

Original data sources are from the **Clustering basic benchmark** repo of the University of Eastern Finland:

[https://cs.joensuu.fi/sipu/datasets/](https://cs.joensuu.fi/sipu/datasets/)

Used/presented in the following *papers*: 

- Fränti, P., Virmajoki, O.: Iterative shrinking method for clustering problems. Pattern Recognition 39 (5), 761--765 (2006).

- Fränti, P., Virmajoki, O., Hautamäki, V.: Fast agglomerative clustering using a k-nearest neighbor graph. IEEE Trans. on Pattern Analysis and Machine Intelligence
28 (11), 1875--1881 (2006).

- Fränti, P., Sieranoja, S.: K-means properties on six clustering benchmark datasets.  Applied Intelligence 48 (12), 4743--4759 (dec 2018)

- Gionis, A., Mannila, H., Tsaparas, P.: Clustering aggregation. ACM Trans. on Know. Disc. from Data (TKDD) 1 (1), 4--es (2007)

- Kärkkäinen, I., Fränti, P.: Gradual model generator for single-pass clustering. Pattern Recognition 40 (3), 784--795 (2007)

- Rezaei, M., Fränti, P.: Can the number of clusters be determined by external indices? IEEE Access 8, 89239--89257 (2020)

Also obtained from the scikit-learn data generation tools:

[https://scikit-learn.org/stable/datasets.html](https://scikit-learn.org/stable/datasets.html) 

*Paper*: Pedregosa, F. et al.: Scikit-learn: Machine learning in Python. Jour. of Machine Learning Research 12, 2825--2830 (2011)

## Multi-dim experiments

        $ python3 tests_Md.py 

Datasets have been generated with **MDCgen**:

[https://www.mathworks.com/matlabcentral/fileexchange/71871-mdcgen-v2/](https://www.mathworks.com/matlabcentral/fileexchange/71871-mdcgen-v2/)

[https://github.com/CN-TU/mdcgen-matlab](https://github.com/CN-TU/mdcgen-matlab)

*Paper*: Iglesias, F., Zseby, T., Ferreira, D., Zimek, A.: Mdcgen: Multidimensional dataset generator for clustering. Jour. of Classiffcation 36 (3), 599--618 (2019).

## Real-data experiments

        $ python3 test_sirena.py dataReal/sirena.csv  

        $ python3 test_mawi.py dataReal/mawi_sample.csv  

Figures are created within the [plots] folder.

- MAWI data [https://mawi.wide.ad.jp/mawi/samplepoint-F/2022/202207311400.html](https://mawi.wide.ad.jp/mawi/samplepoint-F/2022/202207311400.html)
*Paper*: Cho, K., Mitsuya, K., Kato, A.: Traffic data repository at the wide project. Proceedings of the Annual Conference on USENIX Annual Technical Conference. p. 51. ATEC '00, USENIX Association, USA (2000)

- Sirena data [https://upcsirena.app.dexma.com/](https://upcsirena.app.dexma.com/) (Rectorat building, electricity, from 01.01.2022 to 31.12.2022).
*Paper*: Ruiz Martorell, G., López Plazas, F., Cuchí Burgos, A.: Sistema d'informació del consum d'energia i d'aigua de la UPC (Sirena). 1r Congrés UPC Sostenible (2007)

## Parameter sensitivity analysis

        $ python3 pamse2d.py <option>

Options: 'zeta', 'chi', 'chi_min', 'chi_prop', 'e', 'smooth_f', and 'hbs'. ('smooth_f' and 'hbs' correspond to enhancements under study not included in the paper or used in the experiments either).

## Example of SDOclust in update modus

        $ python3 update_test.py

## Generate Critical diagrams

From the [cddiag] folder:

        $ python3 call_diagram.py ../tables/table2d.csv ../tables/tableMd.csv 

Critical diagrams for Silhouette and ARI indices will be created in the same folder (.png files).
