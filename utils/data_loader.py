import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EmbodiedControlDataset(Dataset):
    """
    数据集类，用于加载具身操控任务的数据
    """
    def __init__(self, data_dir, task_names, is_training=True):
        """
        初始化数据集
        
        Args:
            data_dir: 数据根目录
            task_names: 任务名称列表
            is_training: 是否为训练数据
        """
        self.data_dir = data_dir
        self.task_names = task_names
        self.is_training = is_training
        self.data_folder = "train_data" if is_training else "test_data"
        
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """
        加载所有样本数据的路径和基本信息
        """
        samples = []
        for task_name in self.task_names:
            task_dir = os.path.join(self.data_dir, self.data_folder, task_name)
            
            # 加载观察数据和指令数据
            obs_path = os.path.join(task_dir, f"task_obs_data.npy")
            instr_path = os.path.join(task_dir, f"task_instr_data.npy")
            
            # 如果是训练数据，也加载动作数据
            action_path = None
            if self.is_training:
                action_path = os.path.join(task_dir, f"task_action_data.npy")
            
            # 获取样本数量
            obs_data = np.load(obs_path, mmap_mode='r')
            num_samples = obs_data.shape[0]
            
            # 将每个样本的信息添加到列表中
            for i in range(num_samples):
                sample = {
                    "task_name": task_name,
                    "obs_path": obs_path,
                    "instr_path": instr_path,
                    "action_path": action_path,
                    "sample_idx": i
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取单个样本数据
        """
        sample_info = self.samples[idx]
        task_name = sample_info["task_name"]
        sample_idx = sample_info["sample_idx"]
        
        # 加载观察数据 (N, 4, 2, 3, 256, 256)
        obs_data = np.load(sample_info["obs_path"], mmap_mode='r')
        rgb_data = obs_data[sample_idx, :, 0]  # RGB数据，形状为(4, 3, 256, 256)
        pcd_data = obs_data[sample_idx, :, 1]  # 点云数据，形状为(4, 3, 256, 256)
        
        # 加载指令数据 (N, 53, 512)
        instr_data = np.load(sample_info["instr_path"], mmap_mode='r')
        instruction = instr_data[sample_idx]  # 形状为(53, 512)
        
        # 转换为PyTorch张量
        rgb_tensor = torch.tensor(rgb_data, dtype=torch.float32)
        pcd_tensor = torch.tensor(pcd_data, dtype=torch.float32)
        instr_tensor = torch.tensor(instruction, dtype=torch.float32)
        
        # 如果是训练数据，也加载动作数据
        if self.is_training and sample_info["action_path"] is not None:
            action_data = np.load(sample_info["action_path"], mmap_mode='r')
            action = action_data[sample_idx]  # 形状为(1, 8)
            action_tensor = torch.tensor(action, dtype=torch.float32)
            
            return {
                "task_name": task_name,
                "rgb": rgb_tensor,
                "pcd": pcd_tensor,
                "instruction": instr_tensor,
                "action": action_tensor
            }
        else:
            return {
                "task_name": task_name,
                "rgb": rgb_tensor,
                "pcd": pcd_tensor,
                "instruction": instr_tensor
            }

def get_dataloader(data_dir, task_names, batch_size=16, is_training=True, num_workers=4, shuffle=True):
    """
    创建数据加载器
    
    Args:
        data_dir: 数据根目录
        task_names: 任务名称列表
        batch_size: 批次大小
        is_training: 是否为训练数据
        num_workers: 数据加载的工作线程数
        shuffle: 是否打乱数据
    
    Returns:
        数据加载器
    """
    dataset = EmbodiedControlDataset(data_dir, task_names, is_training)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader