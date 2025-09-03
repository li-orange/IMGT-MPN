# IMGT-MPNN
IMGT-MPNN is a molecular graph neural network framework for molecule property prediction. This README provides an overview of the workflow, including environment setup, quick start, configuration files, datasets, and how to use your own data.

## 1. Environment Setup
We recommend starting from a fresh conda/micromamba environment with Python 3.8.

👉 Please follow the detailed [INSTALL.md](INSTALL.md) guide for the step-by-step installation process, including required packages and the correct installation order.

## 2. Run Main Program
Once the virtual environment is activated, run the main program:
	 `python main_molecules_graph_regression.py `
The program will load the dataset based on the configuration file and start training.

## 3. Modify Configuration Files
### 3.1 Modify config File
Example configuration file path: config/molecules_graph_regression_GatedGCN_TOX21_6k.json

The configuration file includes settings for GPU, dataset, training parameters, model parameters, etc.

### 3.2 Other Configuration
You can create your own configuration files as needed, ensuring the paths and formats are correct.

## 4. Add New Dataset
### 4.1 Create Dataset Processing File
In the  `prepare_molecules_xxx.ipynb ` file, create a new dataset processing script and modify the MoleculeDatasetDGL class to support the new dataset.

### 4.2 Modify LoadData Function
In the  `LoadData ` function, ensure that the paths and data processing methods for the new dataset are correct.

### 4.3 Modify MoleculeDataset Class
In the  `MoleculeDataset ` class, add support for the new dataset and ensure that it is properly processed and loaded.

Once training is complete, the dataset will be saved in pkl file format.

### 4.4 Call New Dataset via Configuration File
In the configuration file, modify the datasets field to include the new dataset name:
	 `"datasets": ["NewDataset"] `
## 5. Model
The model code is located in the  `main_molecules_graph_regression.py ` file. This file is responsible for loading data, configuring the model, training the model, and producing outputs.

You can adjust the model structure as needed, such as modifying the number of layers or adding new layers.

## 6. Notes
If you encounter out-of-memory errors, try reducing the batch size (batch_size) or using a different GPU.

Ensure the paths and settings in the configuration file are correct to avoid errors when loading data or the model.
