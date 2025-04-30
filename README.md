# DecoupleNET---UCL-BIOC0025-MSci-Project
This repository is a submission for UCL module BIOC0025 and contains all relevant files pertaining to my work in the Hansen Lab investigating the relationship between the performance and architecture of neural networks for virtual decoupling of protein C13 NMR spectra.

The `Lab Book.ipynb` Jupyter Notebook contains an indepth summary of all of my work, experiments, and data collected throughout the year.

Several cells in the `Lab Book.ipynb` import files from the `saved_model` and `experiment_scripts` directories, therefore it is recommended that this whole repository is downloaded rather than individual files.  

All .py scripts should be run in a virtual environment with Python 3.10.1 and all of the library versions listed in the `requirements.txt` file.

The .py scripts that have the `DecoupleNET` prefix are standalone scripts which can be used to run random predictions on a pre-trained model or train a new model. 

The .py scripts that have the `predictions` suffix are standalone scripts which can be used only to view predictions on pre-trained models.

NB: All pre-trained models are stored in the `saved_model` directory, hence it must also be downloaded for the .py scripts to be able to load the pre-trained models. 
