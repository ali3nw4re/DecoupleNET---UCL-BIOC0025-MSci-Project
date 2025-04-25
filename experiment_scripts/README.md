This folder contains simple .py files which were used to generate all of the validation MSE loss data. 
These scripts cannot plot verification examples or generate a new dataset, they load in a pre-generated dataset, stored as a .csv file.

New .csv files can be made using any of the original DecoupleNET scripts, found in the main directory.
The original scripts must be edited such that they include the line:

`training_df.to_csv("EXPERIMENT_TRAINING_SET_10K_1.csv", index=False)`

Ideally, above the line:

`x_coupled = training_df["Coupled"]`
