环境准备指南
为了保证代码可复现，推荐使用 micromamba 搭建环境（比 conda/mamba 更轻量，安装快且国内更稳定）。

1. 安装 micromamba
官方文档：micromamba installation

Linux (x86_64)

# 下载并解压
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

# 初始化（写入到 ~/.bashrc）
./bin/micromamba shell init -s bash -r ~/micromamba

# 刷新配置（或直接重启终端）
source ~/.bashrc
验证安装

micromamba --version
（可选）使用 mamba 命令别名

echo "alias mamba=micromamba" >> ~/.bashrc
source ~/.bashrc
2. 配置镜像（推荐清华源）
编辑配置文件：

vim ~/micromamba/.mambarc
示例内容：

channels:
  - conda-forge
  - defaults

channel_priority: strict
刷新配置：

source ~/.bashrc
3. 创建环境
在项目根目录下执行：

micromamba create -n IMGT-MPN -f environment_gpu.yml
micromamba activate IMGT-MPN
4. 安装 pip 部分依赖
⚠️ 注意：micromamba 不会自动安装 environment.yml 里的 - pip: 部分，需要手动执行。

4.1 安装 PyTorch (CUDA 11.1 对应版本)

pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html
4.2 验证 CUDA 是否可用

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
4.3 安装 DGL（cu111 版本）

pip install dgl-cu111==0.9.1.post1 -f https://data.dgl.ai/wheels/repo.html
4.4 安装其他依赖

pip install tensorboard==2.16.2
pip install ogb==1.3.6 torch-geometric==2.1.0 torch-scatter==2.0.9 \
    torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1
pip install dgllife==0.3.2
5. 验证安装
运行以下命令，确保关键库都能正常导入：

python - <<'PY'
import torch, dgl, torch_geometric, rdkit, deepchem
print("torch:", torch.__version__, "cuda?", torch.cuda.is_available(), "cuda", torch.version.cuda)
print("dgl:", dgl.__version__)
print("torch-geometric:", torch_geometric.__version__)
print("rdkit:", rdkit.__version__)
print("deepchem:", deepchem.__version__)
PY
如果能正常输出版本号，说明环境已配置成功 🎉。
