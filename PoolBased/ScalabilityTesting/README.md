# Broader Applications.
This folder contains the code used for pool-based Bayesian optimization of a separate dataset to test the scalability of the batch Bayesian optimization frameworks from this study. We tested our BBO framework on the experimental dataset reported by Gongora et al. in the Journal of Science Advances.[1] This dataset contains 3D printed crossbarrel part geometry and mechanical properties that was collected in a grid search approach. The dataset has 600 samples with three repeats for a total of 1800 rows of information, which represents a larger and different dataset to the one collected in our study. The input variables are:  number of struts, displacement angle (deg), strut radius (mm), and strut thickness (mm). These are referred to as n, theta, r, and t, respectively. The outputs in is the toughness of structures (J or J/m3). 

### References: 
1. Aldair E. Gongora et al. ,A Bayesian experimental autonomous researcher for mechanical design.Sci. Adv.6,eaaz1708(2020).DOI: DOI:10.1126/sciadv.aaz1708






