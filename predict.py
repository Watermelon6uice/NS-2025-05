import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json

from models.multimodal_model import MultimodalEmbodiedModel
from utils.data_loader import get_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='测试具身操控模型并生成预测')
    
    parser.add_argument('--data_dir', type=str, default='./NS-2025-11-data', 
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./predictions', 
                        help='预测结果输出目录')
    parser.add_argument('--model_path', type=str, default='./output/best_model.pth', 
                        help='模型权重路径')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='批次大小')
    parser.add_argument('--feature_dim', type=int, default=768, 
                        help='特征维度')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='数据加载线程数')
    
    return parser.parse_args()

def compute_metrics_with_ground_truth(pred, target, pos_threshold=0.10, rot_threshold=0.25):
    """
    如果有真实标签，计算评估指标
    
    Args:
        pred: np.ndarray - 预测的动作，形状为(N, 1, 8)
        target: np.ndarray - 真实动作，形状为(N, 1, 8)
        pos_threshold: 位置误差阈值，单位为米
        rot_threshold: 旋转误差阈值，单位为弧度
    
    Returns:
        metrics: 评估指标字典
    """
    # 分离位置、旋转和夹爪状态
    pred = pred.squeeze(1)  # (N, 8)
    target = target.squeeze(1)  # (N, 8)
    
    pred_pos = pred[:, :3]  # (N, 3)
    pred_rot = pred[:, 3:7]  # (N, 4)
    pred_gripper = (pred[:, 7:] > 0.5).astype(np.float32)  # (N, 1) 二值化
    
    target_pos = target[:, :3]  # (N, 3)
    target_rot = target[:, 3:7]  # (N, 4)
    target_gripper = target[:, 7:]  # (N, 1)
    
    # 计算位置误差（欧几里得距离）
    pos_error = np.sqrt(np.sum((pred_pos - target_pos) ** 2, axis=1))  # (N,)
    pos_accuracy = np.mean(pos_error < pos_threshold)
    
    # 计算旋转误差（四元数角度）
    # 规范化四元数
    pred_rot_norm = pred_rot / (np.linalg.norm(pred_rot, axis=1, keepdims=True) + 1e-8)
    target_rot_norm = target_rot / (np.linalg.norm(target_rot, axis=1, keepdims=True) + 1e-8)
    
    # 计算四元数点积
    dot_product = np.sum(pred_rot_norm * target_rot_norm, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算旋转角度误差（弧度）
    rot_error = 2 * np.arccos(np.abs(dot_product))  # (N,)
    rot_accuracy = np.mean(rot_error < rot_threshold)
    
    # 计算夹爪状态准确率
    gripper_accuracy = np.mean(np.abs(pred_gripper - target_gripper) < 0.5)
    
    metrics = {
        'traj_pos_acc_0.10': float(pos_accuracy),
        'traj_rot_acc_0.25': float(rot_accuracy),
        'traj_gripper': float(gripper_accuracy),
        'pos_error_mean': float(np.mean(pos_error)),
        'rot_error_mean': float(np.mean(rot_error)),
    }
    
    return metrics

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
    else:
        device = torch.device("cpu")
        print("警告: 未检测到GPU，将使用CPU进行推理，可能会非常慢！")
    
    # 任务列表
    task_names = [
        'put_money_in_safe',
        'stack_blocks',
        'light_bulb_in',
        'put_item_in_drawer',
        'place_cups'
    ]
    
    # 创建测试数据加载器
    test_loader = get_dataloader(
        args.data_dir, 
        task_names, 
        batch_size=args.batch_size, 
        is_training=False, 
        num_workers=args.num_workers,
        shuffle=False  # 不打乱数据，保持原始顺序
    )
    
    # 创建模型并加载权重
    model = MultimodalEmbodiedModel(feature_dim=args.feature_dim)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 处理模型权重 - 如果是从DataParallel保存的
    if list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
        # 创建新的OrderedDict移除'module.'前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model = model.to(device)
    model.eval()
    
    # 如果有多个GPU，使用DataParallel可以加速批处理
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行数据并行推理")
        model = torch.nn.DataParallel(model)
    
    print(f"模型已从 {args.model_path} 加载")
    if 'epoch' in checkpoint:
        print(f"模型已训练 {checkpoint['epoch']} 个epoch")
    if 'metrics' in checkpoint:
        print(f"模型验证指标: {checkpoint['metrics']}")
    
    # 用于存储每个任务的预测结果
    task_predictions = {task: [] for task in task_names}
    task_indices = {task: [] for task in task_names}
    
    # 在测试集上进行预测
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="预测中")):
            # 获取数据
            task_names_batch = batch['task_name']
            rgb = batch['rgb'].to(device, non_blocking=True)
            pcd = batch['pcd'].to(device, non_blocking=True)
            
            # 正确处理指令张量 - 这是一个形状为 (batch_size, 53, 512) 的张量
            instructions = batch['instruction'].to(device, non_blocking=True)
            
            # 前向传播
            action_pred = model(rgb, pcd, instructions)  # 输出形状为 (B, 1, action_dim)
            
            # 将预测结果存储到对应任务的列表中
            for i, task_name in enumerate(task_names_batch):
                # 计算在批次中的全局索引
                global_idx = batch_idx * args.batch_size + i
                task_indices[task_name].append(global_idx)
                
                # 直接存储预测结果而不进行任何维度操作，保持其原有的 (1, action_dim) 形状
                task_predictions[task_name].append(action_pred[i].cpu().numpy())  # 这里的形状应该是 (1, 8)
    
    # 按任务保存预测结果，并尝试与示例的ground truth进行比较
    all_metrics = {}
    
    for task_name in task_names:
        if task_predictions[task_name]:
            # 将预测结果转换为NumPy数组 - 由于每个预测已经是(1, action_dim)形状，
            # 所以简单地堆叠它们应该会得到(N, 1, action_dim)
            predictions = np.array(task_predictions[task_name])
            
            # 保存为.npy文件
            output_path = os.path.join(args.output_dir, f'{task_name}_action.npy')
            np.save(output_path, predictions)
            print(f"{task_name} 的预测结果已保存至 {output_path}")
            print(f"形状: {predictions.shape}")
            
            # 尝试加载示例数据作为参考，进行指标评估
            example_path = os.path.join(args.data_dir, 'example', f'{task_name}_action.npy')
            if os.path.exists(example_path):
                try:
                    example_actions = np.load(example_path)
                    # 确保形状匹配
                    if example_actions.shape == predictions.shape:
                        metrics = compute_metrics_with_ground_truth(predictions, example_actions)
                        all_metrics[task_name] = metrics
                        print(f"{task_name} 的评估指标:")
                        print(f"  位置准确率 (0.10m): {metrics['traj_pos_acc_0.10']:.4f}")
                        print(f"  旋转准确率 (0.25rad): {metrics['traj_rot_acc_0.25']:.4f}")
                        print(f"  夹爪准确率: {metrics['traj_gripper']:.4f}")
                    else:
                        print(f"{task_name} 的形状不匹配: predictions {predictions.shape}, example {example_actions.shape}")
                except Exception as e:
                    print(f"比较 {task_name} 的示例数据时出错: {e}")
    
    # 保存所有指标结果
    if all_metrics:
        # 计算平均指标
        avg_metrics = {
            'traj_pos_acc_0.10': np.mean([metrics['traj_pos_acc_0.10'] for metrics in all_metrics.values()]),
            'traj_rot_acc_0.25': np.mean([metrics['traj_rot_acc_0.25'] for metrics in all_metrics.values()]),
            'traj_gripper': np.mean([metrics['traj_gripper'] for metrics in all_metrics.values()]),
        }
        all_metrics['average'] = avg_metrics
        
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"评估指标已保存至 {metrics_path}")
        
        print("\n平均评估指标:")
        print(f"位置准确率 (0.10m): {avg_metrics['traj_pos_acc_0.10']:.4f}")
        print(f"旋转准确率 (0.25rad): {avg_metrics['traj_rot_acc_0.25']:.4f}")
        print(f"夹爪准确率: {avg_metrics['traj_gripper']:.4f}")
    
    print("预测完成!")

if __name__ == "__main__":
    main()