# Environment Setup Guide
To ensure reproducibility, we recommend using micromamba to set up the environment (lighter than conda/mamba, faster to install, and more stable in China).

## 1. Install micromamba
Official documentation: [micromamba installation](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
### Download and extract
```bash
# Linux (x86_64)
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
```
### Initialize (writes to ~/.bashrc)
```bash
./bin/micromamba shell init -s bash -r ~/micromamba
```
### Refresh configuration (or restart the terminal)
```bash
source ~/.bashrc
```
### Verify installation
```bash
micromamba --version
```
### (Optional) Alias mamba to micromamba
```bash
echo "alias mamba=micromamba" >> ~/.bashrc
source ~/.bashrc
```
## 2. Configure mirror
Edit the configuration file:
```bash
vim ~/micromamba/.mambarc
```
```bash
# Example content:
channels:
  - conda-forge
  - defaults

channel_priority: strict
```
### Refresh configuration:
```bash
source ~/.bashrc
```
## 3. Create the environment
From the project root directory:
```bash
micromamba create -n IMGT-MPN -f environment_gpu.yml
micromamba activate IMGT-MPN
```
## 4. Install pip dependencies
⚠️ Note: micromamba does not automatically install the - pip: section of environment.yml. You need to run pip commands manually.

### 4.1 Install PyTorch (CUDA 11.1 version)
```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```
### 4.2 Verify CUDA availability
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```
### 4.3 Install DGL (cu111 version)
```bash
pip install dgl-cu111==0.9.1.post1 -f https://data.dgl.ai/wheels/repo.html
```
### 4.4 Install other dependencies
```bash
pip install ogb==1.3.6 torch-geometric==2.1.0 torch-scatter==2.0.9 \
    torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1
pip install dgllife==0.3.2
```
## 5. Verify installation
Run the following command to make sure all key libraries can be imported:

```bash
python - <<'PY'
import torch, dgl, torch_geometric, rdkit, deepchem
print("torch:", torch.__version__, "cuda?", torch.cuda.is_available(), "cuda", torch.version.cuda)
print("dgl:", dgl.__version__)
print("torch-geometric:", torch_geometric.__version__)
print("rdkit:", rdkit.__version__)
print("deepchem:", deepchem.__version__)
PY
```
If the version numbers are printed correctly, the environment has been successfully set up 🎉.
