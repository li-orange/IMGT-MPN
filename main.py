




"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

from pathlib import Path

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self




"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.load_net import gnn_model # import all GNNS
from data.data import LoadData # import dataset

from itertools import cycle, islice


"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device



def load_datasets(args, config):
    """
    加载多个数据集并返回数据集对象列表，同时返回每个数据集对应的任务数量列表。
    参数:
        args: 命令行参数
        config: 配置文件内容
    返回:
        datasets: 带有索引的加载数据集对象列表
        task_counts: 对应数据集任务数量的列表
    """
    # 动态解析数据集名称
    if args.datasets:  # 如果通过命令行指定了 --datasets 参数
        dataset_names = args.datasets.split(",")  # 按逗号分隔解析多个数据集名称
    else:
        # 从配置文件中加载
        if isinstance(config['datasets'], list):  # 如果配置文件中是列表
            dataset_names = config['datasets']
        else:  # 如果配置文件中是单个字符串
            dataset_names = [config['datasets']]

    # 打印即将加载的数据集信息
    print(f"即将加载的数据集：{dataset_names}")

    # 定义每个数据集名称与任务数量的映射
    dataset_task_mapping = {
        "BACE": 1,
        "BBBP": 1,
        "ClinTox": 2,
        "Sider": 27,
        "TOX21": 12,
        # 可以根据需要继续扩展其他数据集
    }

    # 获取任务数量列表
    task_counts = [dataset_task_mapping.get(dataset_name, 0) for dataset_name in dataset_names]

    # 加载数据集并分配索引
    try:
        datasets = [
            LoadData(dataset_name, dataset_idx=idx) 
            for idx, dataset_name in enumerate(dataset_names)
        ]
        print(f"成功加载数据集：{dataset_names}")
    except Exception as e:
        print(f"加载数据集时出错：{e}")
        raise

    return datasets, task_counts


def preprocess_dataset(dataset, model_name, net_params, t0):
    """
    根据模型需求动态处理数据集，包括添加自环和位置编码
    参数:
        dataset: 数据集对象
        model_name: 模型名称
        net_params: 模型参数配置
        t0: 计时器起点
    """
        # 添加自环
    if model_name in ['GCN', 'GAT'] and net_params.get('self_loop', False):
        print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
        dataset._add_self_loops()

    # 添加位置编码
    if model_name in ['GatedGCN','GCNATT', 'PNA', 'MPNN'] and net_params.get('pos_enc', False):
        print("[!] Adding graph positional encoding.")
        dataset._add_positional_encodings(net_params['pos_enc_dim'])
        print('Time PE:', time.time() - t0)


def interleaved_data_loader(data_loaders):
    """
    创建一个交替加载器，从多个 DataLoader 中交替加载批次。
    如果所有 DataLoader 都被耗尽过，则停止。

    参数:
        data_loaders: 一个包含多个 DataLoader 的列表。
    """
    exhausted_loaders = [False] * len(data_loaders)  # 标记每个 DataLoader 是否耗尽
    consumed_loaders = [False] * len(data_loaders)  # 标记每个 DataLoader 是否已经被完全消费
    data_loaders_iter = [iter(dl) for dl in data_loaders]  # 初始化每个 DataLoader 的迭代器

    while True:
        all_exhausted = True  # 用于检查所有 DataLoader 是否耗尽
        for idx, loader_iter in enumerate(data_loaders_iter):
            if consumed_loaders[idx]:
                continue  # 如果当前 DataLoader 已经被完全消费，跳过
            try:
                yield next(loader_iter), idx
                all_exhausted = False  # 如果还可以加载数据，则标记为未耗尽
            except StopIteration:
                # 当前 DataLoader 耗尽，标记为已耗尽
                print(f"DataLoader {idx} 已耗尽，重新初始化。")
                exhausted_loaders[idx] = True
                # 重新初始化当前 DataLoader，准备下次继续
                data_loaders_iter[idx] = iter(data_loaders[idx])
                
                # 标记为已经消费过
                consumed_loaders[idx] = True

        # 如果所有 DataLoader 都已消费过，则退出训练
        if all(exhausted_loaders):
            print("所有数据集都已经被完全消费，训练结束。")
            break

  




"""
    VIEWING MODEL CONFIG AND PARAMS
"""
def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

"""
    TRAINING CODE
"""

def train_val_pipeline_interleaved(MODEL_NAME, datasets, params, net_params, dirs):
    """
    每个批次交替训练多个数据集。
    """
    t0 = time.time()
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # 数据预处理
    for dataset in datasets:
        preprocess_dataset(dataset, MODEL_NAME, net_params, t0)

    # 创建 DataLoader
    data_loaders = [
        DataLoader(dataset.train, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
        for dataset in datasets
    ]
    val_loaders = [
        DataLoader(dataset.val, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
        for dataset in datasets
    ]
    test_loaders = [
        DataLoader(dataset.test, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
        for dataset in datasets
    ]

    # 初始化模型和优化器
    model = gnn_model(MODEL_NAME, net_params).to(device)
    if False:
        # If you are continuing training from a saved model (e.g., from the previous dataset)
        prev_model_path = 'benchmarking-gnns-master/out/molecules_graph_regression/checkpoints/GatedGCN_ClinTox_GPU1_12h41m01s_on_Dec_25_2024/best_model_dataset_0.pth'  # Path to the model you want to load
        if os.path.exists(prev_model_path):
            print(f"Loading model weights from {prev_model_path}")
            model.load_state_dict(torch.load(prev_model_path))
        else:
            print(f"Starting with a new model, no weights loaded.")

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])

    # 早停变量
    patience_counters = [0] * len(datasets)
    best_val_auc = [0] * len(datasets)
    best_test_auc = [0] * len(datasets)
    best_val_rmse = [100] * len(datasets)
    best_test_rmse = [100] * len(datasets)

    from train.train_molecules_graph_regression import train_epoch_sparse_interleaved as train_epoch, evaluate_network_sparse_new as evaluate_network
    for epoch in range(params['epochs']):
        print(f"\nEpoch {epoch + 1}/{params['epochs']}")

        # 创建交替加载器
        interleaved_loader = interleaved_data_loader(data_loaders)

        # 每个批次交替训练
        epoch_loss, train_rmse, optimizer, train_auc = train_epoch(
            model, optimizer, device, interleaved_loader, net_params['num_tasks'], net_params['task_type']
        )
        print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Train AUC: {train_auc:.4f}, Train RMSE: {train_rmse:.4f}")

        # 验证和测试
        for idx, (val_loader, test_loader) in enumerate(zip(val_loaders, test_loaders)):
            val_loss, val_rmse , val_auc = evaluate_network(model, device, val_loader, net_params['num_tasks'], net_params['task_type'], dataset_idx=idx, test_val_flag='val')
            test_loss, test_rmse, test_auc = evaluate_network(model, device, test_loader, net_params['num_tasks'], net_params['task_type'], dataset_idx=idx, test_val_flag='test')

            print(f"Dataset {idx}: Val Loss: {val_loss:.4f}, Val AUC: {val_auc if val_auc else 'N/A'}, Val RMSE: {val_rmse if val_rmse else 'N/A'}")
            print(f"Dataset {idx}: Test Loss: {test_loss:.4f}, Test AUC: {test_auc if test_auc else 'N/A'}, Test RMSE: {test_rmse if test_rmse else 'N/A'}")

            if net_params['task_type'] == 'cls':
                # 更新早停计数器
                if val_auc and val_auc > best_val_auc[idx]:
                    best_val_auc[idx] = val_auc
                if test_auc and test_auc > best_test_auc[idx]:
                    best_test_auc[idx] = test_auc
                    patience_counters[idx] = 0  # 重置耐心计数器
                else:
                    patience_counters[idx] += 1

                # 保存最佳模型
                if test_auc and test_auc == best_test_auc[idx]:
                    model_path = os.path.join(root_ckpt_dir, f"best_model_dataset_{idx}.pth")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save(model.state_dict(), model_path)
                    print(f"保存最佳模型: {model_path}")

            elif net_params['task_type'] == 'reg':
                # 更新早停计数器
                if val_rmse and val_rmse < best_val_rmse[idx]:
                    best_val_rmse[idx] = val_rmse
                if test_rmse and test_rmse < best_test_rmse[idx]:
                    best_test_rmse[idx] = test_rmse
                    patience_counters[idx] = 0  # 重置耐心计数器
                else:
                    patience_counters[idx] += 1

                # 保存最佳模型
                if test_rmse and test_rmse == best_test_rmse[idx]:
                    model_path = os.path.join(root_ckpt_dir, f"best_model_dataset_{idx}.pth")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save(model.state_dict(), model_path)
                    print(f"保存最佳模型: {model_path}")

        # 早停检查：当所有数据集的耐心计数器都达到阈值时，停止训练
        if all(counter >= params['patience'] for counter in patience_counters):
            print("所有数据集的早停条件均已满足，训练提前终止。")
            break

    if net_params['task_type'] == 'cls':
        print("\n训练完成，最佳结果：")
        for idx, dataset in enumerate(datasets):
            print(f"Dataset {dataset.name}: Best Val AUC: {best_val_auc[idx]:.4f}, Best Test AUC: {best_test_auc[idx]:.4f}")
    elif net_params['task_type'] == 'reg':
        print("\n训练完成，最佳结果：")
        for idx, dataset in enumerate(datasets):
            print(f"Dataset {dataset.name}: Best Val RMSE: {best_val_rmse[idx]:.4f}, Best Test RMSE: {best_test_rmse[idx]:.4f}")
    

    print(f"Total training time: {time.time() - t0:.2f}s")



def train_val_pipeline_new(MODEL_NAME, datasets, params, net_params, dirs):
    """
    训练和验证流水线，支持多个数据集交替训练
    参数:
        MODEL_NAME: 模型名称
        datasets: 多个数据集对象列表
        params: 训练参数
        net_params: 网络参数
        dirs: 日志和检查点目录
    """
    t0 = time.time()
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # 数据预处理
    for dataset in datasets:
        preprocess_dataset(dataset, MODEL_NAME, net_params, t0)

    # 创建每个数据集的 DataLoader
    train_loaders = [
        DataLoader(dataset.train, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
        for dataset in datasets
    ]
    val_loaders = [
        DataLoader(dataset.val, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
        for dataset in datasets
    ]
    test_loaders = [
        DataLoader(dataset.test, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
        for dataset in datasets
    ]

    # 初始化模型和优化器
    model = gnn_model(MODEL_NAME, net_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=params['lr_reduce_factor'], patience=params['lr_schedule_patience'], verbose=True
    )

    # 保存最佳模型的性能
    best_val_auc = [0] * len(datasets)
    best_test_auc = [0] * len(datasets)
    patience_counters = [0] * len(datasets)

    # 记录日志
    writer = SummaryWriter(log_dir=root_log_dir)

    # 训练循环
    from train.train_molecules_graph_regression import train_epoch_sparse_new as train_epoch, evaluate_network_sparse_new as evaluate_network
    for epoch in range(params['epochs']):
        print(f"\nEpoch {epoch + 1}/{params['epochs']}")

        # 交替训练多个数据集
        epoch_train_losses = []
        for idx, train_loader in enumerate(train_loaders):
            # 使用 train_epoch 训练当前数据集
            epoch_loss, _, optimizer, train_auc = train_epoch(
                model, optimizer, device, train_loader, epoch, net_params['num_tasks'], net_params['task_type'], idx )
            print(f"Dataset {idx}: Train Loss: {epoch_loss:.4f}, Train AUC: {train_auc:.4f}")
            epoch_train_losses.append(epoch_loss)

        # 验证和测试
        for idx, (val_loader, test_loader) in enumerate(zip(val_loaders, test_loaders)):
            val_loss, _ , val_auc = evaluate_network(model, device, val_loader, net_params['num_tasks'], net_params['task_type'], dataset_idx=idx, test_val_flag='val')
            test_loss, _, test_auc = evaluate_network(model, device, test_loader, net_params['num_tasks'], net_params['task_type'], dataset_idx=idx, test_val_flag='test')

            print(f"Dataset {idx}: Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            print(f"Dataset {idx}: Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")

            # 保存最佳模型
            if test_auc > best_test_auc[idx]:
                best_val_auc[idx] = val_auc
                best_test_auc[idx] = test_auc
                model_path = os.path.join(root_ckpt_dir, f"best_model_dataset_{idx}.pth")
                # 确保目录存在
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"保存最佳模型: {model_path}")
                patience_counters[idx] = 0  # 重置耐心计数器
            else:
                patience_counters[idx] += 1

            # 学习率调度
            scheduler.step(val_loss)

        # 早停检查：当所有数据集的耐心计数器都达到阈值时，停止训练
        if all(counter >= params['patience'] for counter in patience_counters):
            print("所有数据集的早停条件均已满足，训练提前终止。")
            break

    # 输出最终结果
    print("\n训练完成，最佳结果：")
    for idx, dataset in enumerate(datasets):
        print(f"Dataset {dataset.name}: Best Val AUC: {best_val_auc[idx]:.4f}, Best Test AUC: {best_test_auc[idx]:.4f}")

    print(f"Total training time: {time.time() - t0:.2f}s")



def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    t0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    if DATASET_NAME =='BACE':
        dataset_idx = 0
    elif DATASET_NAME =='BBBP':
        dataset_idx = 1
    elif DATASET_NAME =='ClinTox':
        dataset_idx = 2
    
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
            
    if MODEL_NAME in ['GatedGCN','PNA','MPNN']:
        if net_params['pos_enc']:
            print("[!] Adding graph positional encoding.")
            dataset._add_positional_encodings(net_params['pos_enc_dim'])
            print('Time PE:',time.time()-t0)
        
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    if False:
        # If you are continuing training from a saved model (e.g., from the previous dataset)
        prev_model_path = 'benchmarking-gnns-master/out/molecules_graph_regression/checkpoints/dataset_1/best_model_dataset1_BBBP.pth'  # Path to the model you want to load
        if os.path.exists(prev_model_path):
            print(f"Loading model weights from {prev_model_path}")
            model.load_state_dict(torch.load(prev_model_path))
        else:
            print(f"Starting with a new model, no weights loaded.")


    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs = [], [] 
    
    # batching exception for Diffpool
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    
    if MODEL_NAME in ['RingGNN', '3WLGNN']:
        # import train functions specific for WLGNNs
        from train.train_molecules_graph_regression import train_epoch_dense as train_epoch, evaluate_network_dense as evaluate_network
        from functools import partial # util function to pass edge_feat to collate function

        train_loader = DataLoader(trainset, shuffle=True, collate_fn=partial(dataset.collate_dense_gnn, edge_feat=net_params['edge_feat']))
        val_loader = DataLoader(valset, shuffle=False, collate_fn=partial(dataset.collate_dense_gnn, edge_feat=net_params['edge_feat']))
        test_loader = DataLoader(testset, shuffle=False, collate_fn=partial(dataset.collate_dense_gnn, edge_feat=net_params['edge_feat']))
        
    else:
        # import train functions for all other GNNs
        from train.train_molecules_graph_regression import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
        
        train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last, collate_fn=dataset.collate)
        val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
        test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last, collate_fn=dataset.collate)
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        best_train_auc = 0
        best_val_auc = 0 
        best_test_auc = 0
        patience = params['patience']
        counter = 0
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                if MODEL_NAME in ['RingGNN', '3WLGNN']: # since different batch training function for RingGNN
                    epoch_train_loss, epoch_train_mae, optimizer, train_auc = train_epoch(model, optimizer, device, train_loader, epoch, params['batch_size'], dataset_idx)
                else:   # for all other models common train function 
                    epoch_train_loss, epoch_train_mae, optimizer, train_auc = train_epoch(model, optimizer, device, train_loader, epoch, net_params['num_tasks'], net_params['task_type'], dataset_idx )
                    
                epoch_val_loss, epoch_val_mae, val_auc = evaluate_network(model, device, val_loader, epoch, net_params['num_tasks'], net_params['task_type'], dataset_idx, 'val')
                _, epoch_test_mae, test_auc = evaluate_network(model, device, test_loader, epoch, net_params['num_tasks'], net_params['task_type'], dataset_idx, 'test')

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_MAEs.append(epoch_train_mae)           
                epoch_val_MAEs.append(epoch_val_mae)                 

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                #writer.add_scalar('train/_mae', epoch_train_mae, epoch)
                #writer.add_scalar('val/_mae', epoch_val_mae, epoch)
                #writer.add_scalar('test/_mae', epoch_test_mae, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('train__auc', train_auc, epoch)
                writer.add_scalar('val_auc', val_auc, epoch)
                writer.add_scalar('test_auc', test_auc, epoch)

                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,train_AUC=train_auc, 
                              val_AUC=val_auc,test_AUC=test_auc)
                """
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_MAE=epoch_train_mae, val_MAE=epoch_val_mae,
                              test_MAE=epoch_test_mae, train_AUC=train_auc, val_AUC=val_auc,
                              test_AUC=test_auc)
                """

                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                
                root_ckpt_dir_dataset = 'benchmarking-gnns-master/out/molecules_graph_regression/checkpoints'
                ckpt_dir_dataset = os.path.join(root_ckpt_dir_dataset, "dataset_" + str(dataset_idx))
                if not os.path.exists(ckpt_dir_dataset):
                    os.makedirs(ckpt_dir_dataset)

                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
                
                if train_auc > best_train_auc:
                    best_train_auc = train_auc
                if val_auc > best_val_auc:
                    best_val_auc = val_auc

                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    counter = 0  # 重置耐心计数器
                    print(f"保存最新模型: {ckpt_dir_dataset}:best_model_dataset{str(dataset_idx)}_{DATASET_NAME}")
                    torch.save(model.state_dict(), '{}.pth'.format(ckpt_dir_dataset + "/best_model_dataset"+str(dataset_idx)+"_"+ DATASET_NAME))  # 保存最优模型
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"早停法激活，第 {epoch} 轮停止训练")
                        break
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    
    _, test_mae, Test_auc = evaluate_network(model, device, test_loader, epoch, net_params['num_tasks'], net_params['task_type'], dataset_idx, 'test')
    _, train_mae, Train_auc = evaluate_network(model, device, val_loader, epoch, net_params['num_tasks'], net_params['task_type'], dataset_idx, 'val')
    #输出最佳auc
    print("Best train_AUC: {:.4f} Best test_auc: {:.4f} Best val_auc: {:.4f}".format(best_train_auc,best_test_auc,best_val_auc))
    print("Test MAE: {:.4f}".format(test_mae))
    print("Train MAE: {:.4f}".format(train_mae))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_mae, train_mae, epoch, (time.time()-t0)/3600, np.mean(per_epoch_time)))
        

def test_model(model, test_loader, dataset_idx, net_params, device):
    from train.train_molecules_graph_regression import train_epoch_sparse as train_epoch, evaluate_network_sparse as evaluate_network
    _, test_mae, Test_auc = evaluate_network(model, device, test_loader, 0, net_params['num_tasks'], net_params['task_type'], dataset_idx, 'test')
    print(f"Dataset {dataset_idx} Test Accuracy: {Test_auc * 100:.2f}%")

def main():    
    """
        USER CONTROLS
    """
    #    config_file_names:
    #       molecules_graph_classification.json
    #       molecules_graph_regression.json
    this_dir = Path(__file__).resolve().parent
    data_root = this_dir / "configs"
    config_file_path = data_root / 'molecules_graph_regression.json'
    if os.path.exists(config_file_path):
        print("配置文件存在")
    else:
        print("配置文件不存在")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=config_file_path,help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--datasets', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")    
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--pos_enc', help="Please give a value for pos_enc")
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:        
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']

    # Load multiple datasets

    
    # 数据集加载
    datasets, num_tasks = load_datasets(args, config)

    task_type = ''
    DATASET_NAME = ''

    # 打印加载信息
    print("加载的数据集：")
    for idx, dataset in enumerate(datasets):
        DATASET_NAME += dataset.name
        print(f"索引: {idx}, 数据集: {dataset.name}, 任务数量:{num_tasks[idx]}")

    if datasets[0].name in ['TOX21', 'HIV', 'PCBA', 'BACE', 'BBBP', 'ToxCast', 'Sider', 'ClinTox']:
        task_type = 'cls'
    else:
        task_type = 'reg'

    

    """
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']

    if DATASET_NAME in ['TOX21', 'HIV', 'PCBA', 'BACE', 'BBBP', 'ToxCast', 'Sider', 'ClinTox']:
        task_type = 'cls'
    else:
        task_type = 'reg'

    #Bunch of classification tasks
    if DATASET_NAME == "TOX21":
        num_tasks = 12
    elif DATASET_NAME == "PCBA":
        num_tasks = 128
    elif DATASET_NAME == "BACE":
        num_tasks = 1
    elif DATASET_NAME == "HIV":
        num_tasks = 1
    elif DATASET_NAME == "BBBP":
        num_tasks = 1
    elif DATASET_NAME == "ToxCast":
        num_tasks = 617
    elif DATASET_NAME == "Sider":
        num_tasks = 27
    elif DATASET_NAME == "ClinTox":
        num_tasks = 1
    elif DATASET_NAME == 'ESOL':
        num_tasks = 1
    elif DATASET_NAME == 'FreeSolv':
        num_tasks = 1
    elif DATASET_NAME == 'Lipophilicity':
        num_tasks = 1
    elif DATASET_NAME == 'QM7':
        num_tasks = 1
    elif DATASET_NAME == 'QM8':
        num_tasks = 12
    elif DATASET_NAME == 'QM9':
        num_tasks = 12
    else:
        raise ValueError("Invalid dataset name.")
    
    dataset = LoadData(DATASET_NAME)
    """


    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)   
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated=='True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred=='True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat=='True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop=='True' else False
    if args.pos_enc is not None:
        net_params['pos_enc'] = True if args.pos_enc=='True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)
        
    
    # ZINC
    #net_params['num_atom_type'] = dataset.num_atom_type
    #net_params['num_bond_type'] = dataset.num_bond_type

    #num_tasks
    net_params['num_tasks'] = num_tasks
    net_params['task_type'] = task_type

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(MODEL_NAME, net_params)
    train_val_pipeline_interleaved(MODEL_NAME, datasets, params, net_params, dirs)

    
    
    
    
    
    
    
main()    
















