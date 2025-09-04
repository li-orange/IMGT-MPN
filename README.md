# Molecular Property Prediction Based on Improved Graph Transformer Network and Multi-Task Joint Learning Strategy
Thank you for your interest in this database, which serves as a supplement to the paper currently being prepared for publication. 

**IMGT-MPNN** is a molecular property prediction framework designed to capture both local chemical environments and global molecular structures. It combines an improved Graph Transformer network with a multi-task joint learning strategy. 

This **README** provides an overview of the workflow, including environment setup, quick start, configuration files, datasets, and how to use your own data.

## 1. Environment Setup
We recommend creating a fresh conda/micromamba environment with Python 3.8. Two default environment files are provided:

CPU only: `environment_cpu.yml`

GPU (CUDA): `environment_gpu.yml`

Create the environment with:
```bash
micromamba create -f environment_gpu.yml
micromamba activate IMGT-MPNN
```
Please follow the detailed [INSTALL.md](INSTALL.md) guide for the step-by-step installation process, including required packages and the correct installation order.

## 2. Quick Start
Once the virtual environment is activated, run the main program:
```bash
python main.py --config configs/molecules_graph_regression.json
```
This will:
- Load the dataset
- Build the model according to the config
- Start training

You can switch to classification or other configs, e.g.:
```bash
python main.py --config configs/molecules_graph_classification.json
```
## 3. Configuration Files
Configuration files are located in configs/.

Example: configs/molecules_graph_regression.json

They define:

- Device setup (CPU / GPU)
- Dataset (path, preprocessing)
- Model parameters (layers, hidden size, etc.)
- Training parameters (batch size, epochs, learning rate, etc.)

To use your own dataset or model, simply copy and modify a JSON config file.

## 4. Data
The raw molecular datasets are located in `data/molecules/`.

### 4.1 Preprocessed Datasets
The repository already provides `.pkl` files that can be directly loaded for training.

⚠️ Note: The `Lipophilicity` dataset is provided as a compressed archive.
Please unzip it before use:

```bash
unzip Lipophilicity_8-1-1.zip
```
### 4.2 Processing from Scratch
There are 7 **Jupyter Notebooks( `prepare_molecules_xxx.ipynb ` )** in the `data/` directory. 
After installing the virtual environment, you can run them in order.
Each notebook processes one dataset:

1. Reads the original CSV file (SMILES and labels).

2. Converts molecules into **DGL graphs** via the `MoleculeDatasetDGL()` function.

3. Saves the processed dataset as `.pkl` files for training.

### 4.3 Adding a New Dataset
To quickly add your own dataset:

- Create a new Jupyter Notebook under data/, following the structure of existing notebooks.
- Create a folder for your dataset and place the raw CSV file (smiles + labels) inside.
- Use the notebook to test whether SMILES can be correctly read and converted into DGL graphs.
- Save the processed dataset as `.pkl`.
- Register the dataset by modifying:
  - `LoadData` function
  - `MoleculeDatasetDGL()` function
  - `MoleculeDataset` class

This ensures your dataset is integrated in a consistent and standardized way.

### 4.4 Call New Dataset via Configuration File
In the configuration file, modify the datasets field to include the new dataset name:
	 `"datasets": ["NewDataset"] `
  
## 5. Model

Model implementations are located in the `nets/` directory. They include various GNN architectures such as GCN, GAT, GIN, PNA, and MPNN.

The unified model loading function is implemented in:
```bash
nets/load_net.py
```

### 5.1 Adding a New Model
To add a new model:

1. Create a new Python file in `nets/`, e.g. `my_gnn_net.py`.
2. Define your model class inside the file.
3. Update `load_net.py` to include your new model for easy access.

The model code is located in the  `nets/ ` file. This file is responsible for loading data, configuring the model, training the model, and producing outputs.

You can adjust the model structure as needed, such as modifying the number of layers or adding new layers.

## 6. Training
Training scripts are located in the `train/` directory. The most important code file is `train_molecules_graph_regression.py`.


## 7. Notes
If you encounter out-of-memory errors, try reducing the batch size (batch_size) or using a different GPU.

Ensure the paths and settings in the configuration file are correct to avoid errors when loading data or the model.

