
# SDOclust experiments

Dear reviewers,

This is the version of SDOclust (fully operative) used in our tests and experiments as shown in the paper submission. If (and/or after) the paper is accepted for publication, we will transform this current version into a DOI-citable package in figshare to meet best practices of Reproducible Research (RR). This repository will remain, but migth change due to potential updates.

Please, note that [tables], [plots] and [cddiag] folders contain results as provided in the paper. By running the scripts below, files with results will be overwritten.

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

## Multi-dim experiments

        $ python3 tests_Md.py 

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
