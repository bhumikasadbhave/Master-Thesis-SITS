# Master-Thesis-SITS

Sugarbeet Stress Detection using SITS 

This is the code-base for sugar-beet stress detection.
The structure of the code-base is as follows:

Main files:
-- stress_detection_system.py: The script that does the preprocessing, modeling using the 3D_AE_B10 model with temporal encodings and evaluation of sugar-beet images. Can be run directly from command line. 

-- reproducing_results.ipynb: Notebook for reproducing results. It includes 2 sections:
    -- Results from JSONs: Average results over 3 executions, reported for all autoencoder-based models. Since these results are saved, we dont need to load the sentinel-2 data. These numbers are the ones reported in the Manuscript.
    -- Running the saved models: The saved baseline and autoencoder models are loaded and run on evaluation data. These numbers report result over a single execution.

-- Data Preprocessing
    -- scripts: Consists of multiple .py helper files which consist of functions that aid the data-preprocessing.  
    -- Results: Utility JSON files and trained models, these are used to reproduce results.
    -- Pipeline: Consists of the data-preprocessing pipeline which uses multiple fucntions for creating model-ready tensors and data-loaders.

-- Modeling
    -- model_scripts: Scripts that aid feature extraction, model training and other modeling operations
    -- Jupyter notebooks
        -- baseline_models: Step-by-step execution of baseline models- clustering the raw data, histogram features and PCA. 
        -- autoencoder_models: Step-by-step execution of autoencoder models- 2D_AE_B10, 3D_AE_B10 with and without temporal encodings. The executions are performed over 3 runs, and hyper-parameters for training are selected using k-fold cross validation.
        -- MAE_implementation: SatMAE architecture implementation for our use case.
        -- visualisations: For reconstructed images, plots, and generating final deliverables-stress maps.

-- Evaluation
    -- evaluation_scripts: Scripts that aid the evaluation of clustering results and some visualisations.

-- Experimentation
    -- expt_scripts: Scripts for functions that aid the experiments performed and plots.
    -- Jupyter notebooks:
        -- 2024-data-expt: Experimenting with 2024 data, the full stress detection pipeline is run on 2024 data by running the stres_detection_system.py from command line. However, to visualise results, plots and intermediate results, I have performed the stress detection step-by-step as well, in this notebook.
        -- sub-patch-size-expt: For varying the sub-patch size and creating the plots used in the Manuscript.
        -- threshold-expt: For assesing the effct of varying the sub-patch-to-patch threshold and creating the plots.
        -- vi-expt: For using VIs and fewer bands, to see how it influences the clustering results.


