## Master Thesis: Sugarbeet Stress Detection using SITS 🌱📡
------------------------------------------------------------

**Course**: MSc. in Artificial Intelligence  
**Institution**: CAIRO, THWS  
**Date**: 5th August, 2025 

This repository contains the complete codebase for the Master's thesis titled "Sugarbeet Stress Detection using Satellite Image Time Series (SITS)."

The thesis investigates stress detection in sugarbeet crops using Sentinel-2 image time series and various machine learning and deep learning models, with a focus on autoencoder-based architectures. The primary objective is to build a stress detection pipeline that works with minimal supervision, and is generalizable to data from different years.

The stress detection pipeline requires minimal configuration—primarily file path setup—and is capable of direct deployment on new sugar-beet seasons.

Key steps include:
- Preprocessing of raw Sentinel-2 imagery into model-ready tensors
- Division of fields into smaller sub-patches for localized stress detection
- Extraction of spatiotemporal features using a 3D convolutional autoencoder with temporal encodings
- Unsupervised clustering of sub-patches into stressed or healthy categories
- Threshold-based aggregation of sub-patch predictions to produce field-level labels
- Generation of stress maps for fine-grained visual interpretation

This modular, scalable system allows for cross-seasonal deployment by simply updating the config.py file to apply the model to new Sentinel-2 data.

### 🗂 Repository Structure
```  
Master-Thesis-SITS/  
│  
├── stress_detection_system.py          # Main script for full pipeline (preprocessing → model → evaluation)  
├── reproducing_results.ipynb           # Notebook to reproduce the results from manuscript  
│  
├── Data Preprocessing/  
│   ├── scripts/                        # Helper scripts for preprocessing steps  
│   ├── Results/                        # Stored models, metrics, utility JSONs  
│   └── Pipeline/                       # Main data preprocessing pipeline  
│  
├── Modeling/  
│   ├── model_scripts/                  # Feature extraction, training, and model-related utilities  
│   └── Jupyter notebooks/  
│       ├── baseline_models.ipynb       # Raw data clustering, histograms, PCA  
│       ├── autoencoder_models.ipynb    # 2D & 3D autoencoders with/without temporal encodings  
│       ├── MAE_implementation.ipynb    # SatMAE adaptation for stress detection  
│       └── visualisations.ipynb        # Stress maps, reconstructed images, and plots  
│  
├── Evaluation/  
│   └── evaluation_scripts/            # Evaluation and visualization for clustering and AE outputs  
│  
└── Experimentation/  
    ├── expt_scripts/                  # Experimental utilities and plotting scripts  
    └── Jupyter notebooks/  
        ├── 2024-data-expt.ipynb       # Full pipeline on unseen 2024 data  
        ├── sub-patch-size-expt.ipynb  # Patch size variation experiments  
        ├── threshold-expt.ipynb       # Sub-patch-to-patch threshold tuning  
        └── vi-expt.ipynb              # Vegetation Indices (VI) experiments  
```  

### 📌 How to Run
1. Run Full Pipeline (Command Line): This performs preprocessing, modeling using the 3D_AE_B10 architecture with temporal encodings, and evaluation.
    python stress_detection_system.py

2. Reproduce Manuscript Results  
Section 1: Uses saved JSONs (no Sentinel-2 data loading required).  
Section 2: Runs saved models on eval data for single-run performance. All saved models are present in Google drive, download the saved models and accordingly change the paths in the config.py.   
Google Drive path: https://drive.google.com/drive/folders/1Hiw_ZVep3FHMhLzMlI5_tQdLUdBk1xyn?usp=sharing  

### 📁 Dataset
The experiments use Sentinel-2 satellite image time series focused on sugarbeet fields. Due to data storage size, raw Sentinel-2 data is not included in the Github repository. However, preprocessing scripts and model-ready formats are provided to replicate results.

### 📬 Contact
For questions, feedback, or collaboration opportunities:
- Email: sadbhavebhumika21@gmail.com
- LinkedIn: https://www.linkedin.com/in/bhumika05/

