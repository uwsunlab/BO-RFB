# PNNL-ML_for_Organic_Flow_Battery_Materials
Machine learning guided synthesis for organic flow batteries and the processes.
This repositiory contains the code used for the experiment conducted at PNNl and UW for synthesizing organic redox batteries.

![File](figures/Figure1_OfficialStage.png)

The following figure is a diagram that shows how to files are organized
![File](figures/file_structure.png)

# Results 
## Model A:
[Open Model A Results Notebook](Experiment_Results/ModelA_results.ipynb)
![File](figures/ModelA.png)
Plots for Model A can be found under ```Experiment_Results/ModelA_results.ipynb```
The uncertainty plots for Model A were constructed in the same file. (SI)

## Model B:
[Open Model B Results Notebook](Experiment_Results/ModelB_results.ipynb)
![File](figures/ModelB.png)
Plots for Model B can be found under ```Experiment_Results/ModelB_results.ipynb```
The uncertainty plots for Model B were constructed in the same file. (SI)

## Model C: 
[Open Model C Results Notebook](Experiment_Results/ModelC_results.ipynb)
![File](figures/ModelC.png)
Plots for Model C can be found under ```Experiment_Results/ModelC_results.ipynb```
The uncertainty plots for Model C were constructed in the same file. (SI)

## Summary of Results
[Open Summary Results Notebook](Experiment_Results/All_results.ipynb)
![File](figures/ResultsSummary.png)
Model evaluations and comparison plots can be found in ```Experiment_Results/All_results.ipynb```

# Method
In total, four rounds of data collection were conducted and are named: Round1, Round2, Round3, and Round_Redo. 
- [Round 1](Round1): Contains data generated from Latin Hypercube sampling and their respective HPLC samples. 
- [Round 2](Round2): Contains the code to extract yield from HPLC and the Bayesian Optimization in round 2 for Model A, Model B, and Model C, as well as their repective HPLC samples. 
- [Round 3](Round3): Contains the code to extract yield from HPLC and the Bayesian Optimization in round 3 for Model A, Model B, and Model C, as well as their repective HPLC samples. 
- [Round Redo](Round_Redo): Contains selected samples that were resynthesized due to inconsistencies.

## Data Extraction
- Explain about the HPLC data
- Explain about the extraction algorithm and the python
![File](figures/HPLC.png)
