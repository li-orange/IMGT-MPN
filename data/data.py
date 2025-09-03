"""
    File to load dataset based on user control from main file
"""
from data.molecules import MoleculeDataset


def LoadData(DATASET_NAME, dataset_idx):
    """
    根据数据集名称加载对应的 Molecule 数据集
    返回：
        dataset: MoleculeDataset 对象
    """
    supported_datasets = ['TOX21', 'BACE', 'BBBP', 'ClinTox', 'HIV', 'Sider', 'FreeSolv', 'Lipophilicity', 'ESOL', 'QM8']

    if DATASET_NAME not in supported_datasets:
        raise ValueError(f"不支持的数据集: {DATASET_NAME}. 支持的数据集: {supported_datasets}")

    print(f"[I] 加载数据集 {DATASET_NAME}...")
    dataset = MoleculeDataset(DATASET_NAME, dataset_idx)
    print(f"[I] 成功加载数据集: {DATASET_NAME}")
    return dataset
    