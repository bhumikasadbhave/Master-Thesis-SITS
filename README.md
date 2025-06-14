# 🌾 Stress Detection System for Sugar-Beet Fields

This repository contains the codebase for an unsupervised stress detection pipeline designed for identification of stress conditions in sugar-beet crops using temporal satellite imagery and autoencoder-based feature extraction.

## 🚀 Overview

The system integrates temporal feature extraction and clustering to identify stressed regions in sugar-beet fields over a growing season. Key components include:

- 3D Autoencoder for unsupervised feature learning  
- Sinusoidal positional encodings to represent acquisition dates  
- K-Means clustering for discovering patterns in latent space  
- Sub-patch-level analysis and field-level results  

## 🧱 Project Structure
<pre>
<code>
```
├── Data-Preprocessing/
│   ├── Pipeline/                        # Preprocessing pipeline and script
│   ├── scripts/                         # Functions for data loading, preprocessing, visualization
│   └── Jupyter notebook for preprocessing demo
│
├── Evaluation/
│   └── evaluation_scripts/             # Functions for evaluation metrics and result visualizations
│
├── Experimentation/
│   ├── expt_scripts/                   # Functions used for experiments 
│   └── Jupyter notebook for threshold and VI-based experimentation
│
├── Modeling/
│   ├── Results/                        # Model output JSONs
│   └── Jupyter notebooks for main model and baselines
│
├── config.py                           # Configuration paths and parameters
├── stress_detection_script.py         # Main pipeline script for future data
└── README.md
```
</code>
</pre>
