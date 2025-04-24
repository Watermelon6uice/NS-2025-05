import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# 尝试导入自动混合精度训练
try:
    from torch.amp import autocast, GradScaler
    HAS_AMP = torch.cuda.is_available()
except ImportError:
    HAS_AMP = False
    print("警告: 无法导入自动混合精度训练功能。")

# 尝试导入tensorboard，如果不存在则使用一个简单的替代品
try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD = True
except ImportError:
    print("警告: 未找到tensorboard模块，将不会记录训练日志。")
    print("如需完整功能，请安装tensorboard: pip install tensorboard")
    USE_TENSORBOARD = False
    
    # 创建一个简单的SummaryWriter替代品
    class DummySummaryWriter:
        def __init__(self, log_dir=None):
            self.log_dir = log_dir
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            print(f"日志将不会被记录 (tensorboard未安装)")
            
        def add_scalar(self, *args, **kwargs):
            pass  # 什么都不做
            
        def close(self):
            pass  # 什么都不做
    
    SummaryWriter = DummySummaryWriter

from models.multimodal_model import MultimodalEmbodiedModel
from utils.data_loader import get_dataloader

class CustomLoss(nn.Module):
    """
    自定义损失函数，分别计算位置、旋转和夹爪状态的损失，
    以优化评估标准：轨迹位置准确率、轨迹旋转准确率和抓取状态准确率
    """
    def __init__(self, pos_weight=1.0, rot_weight=1.0, gripper_weight=1.0):
        super(CustomLoss, self).__init__()
        self.pos_weight = pos_weight  # 位置损失权重
        self.rot_weight = rot_weight  # 旋转损失权重
        self.gripper_weight = gripper_weight  # 夹爪状态损失权重
        
        self.pos_loss_fn = nn.MSELoss()
        self.rot_loss_fn = nn.MSELoss()  # 四元数MSE损失
        # 替换为BCEWithLogitsLoss，以支持混合精度训练
        self.gripper_loss_fn = nn.BCEWithLogitsLoss()  # 夹爪状态二分类损失
        
    def forward(self, pred, target):
        """
        计算综合损失
        输入:
            pred: (B, 1, 8) - 预测的动作
            target: (B, 1, 8) - 目标动作
        输出:
            loss: 总损失
            pos_loss: 位置损失
            rot_loss: 旋转损失
            gripper_loss: 夹爪状态损失
        """
        B = pred.size(0)
        
        # 分离位置、旋转和夹爪状态
        pred = pred.squeeze(1)  # (B, 8)
        target = target.squeeze(1)  # (B, 8)
        
        pred_pos = pred[:, :3]  # (B, 3)
        pred_rot = pred[:, 3:7]  # (B, 4)
        pred_gripper = pred[:, 7:]  # (B, 1) - 注意这里不再应用sigmoid，BCEWithLogitsLoss会内部处理
        
        target_pos = target[:, :3]  # (B, 3)
        target_rot = target[:, 3:7]  # (B, 4)
        target_gripper = target[:, 7:]  # (B, 1)
        
        # 计算位置损失
        pos_loss = self.pos_loss_fn(pred_pos, target_pos)
        
        # 计算旋转损失
        # 确保四元数是单位向量
        pred_rot = pred_rot / (torch.norm(pred_rot, dim=1, keepdim=True) + 1e-8)
        target_rot = target_rot / (torch.norm(target_rot, dim=1, keepdim=True) + 1e-8)
        
        # 计算四元数点积，点积越大表示四元数越接近
        dot_product = torch.sum(pred_rot * target_rot, dim=1, keepdim=True)
        rot_loss = 1.0 - dot_product * dot_product  # 四元数距离
        rot_loss = rot_loss.mean()
        
        # 计算夹爪状态损失 - 使用BCEWithLogitsLoss，它会在内部应用sigmoid
        gripper_loss = self.gripper_loss_fn(pred_gripper, target_gripper)
        
        # 总损失
        total_loss = (
            self.pos_weight * pos_loss + 
            self.rot_weight * rot_loss + 
            self.gripper_weight * gripper_loss
        )
        
        return total_loss, pos_loss, rot_loss, gripper_loss

def compute_metrics(pred, target, pos_threshold=0.10, rot_threshold=0.25):
    """
    计算评估指标
    
    Args:
        pred: (B, 1, 8) - 预测的动作
        target: (B, 1, 8) - 目标动作
        pos_threshold: 位置误差阈值，单位为米
        rot_threshold: 旋转误差阈值，单位为弧度
    
    Returns:
        metrics: 评估指标字典
    """
    batch_size = pred.size(0)
    
    # 分离位置、旋转和夹爪状态
    pred = pred.squeeze(1).detach().cpu().numpy()  # (B, 8)
    target = target.squeeze(1).detach().cpu().numpy()  # (B, 8)
    
    pred_pos = pred[:, :3]  # (B, 3)
    pred_rot = pred[:, 3:7]  # (B, 4)
    pred_gripper = (pred[:, 7:] > 0.5).astype(np.float32)  # (B, 1) 二值化
    
    target_pos = target[:, :3]  # (B, 3)
    target_rot = target[:, 3:7]  # (B, 4)
    target_gripper = target[:, 7:]  # (B, 1)
    
    # 计算位置误差（欧几里得距离）
    pos_error = np.sqrt(np.sum((pred_pos - target_pos) ** 2, axis=1))  # (B,)
    pos_accuracy = np.mean(pos_error < pos_threshold)
    
    # 计算旋转误差（四元数角度）
    # 规范化四元数
    pred_rot_norm = pred_rot / (np.linalg.norm(pred_rot, axis=1, keepdims=True) + 1e-8)
    target_rot_norm = target_rot / (np.linalg.norm(target_rot, axis=1, keepdims=True) + 1e-8)
    
    # 计算四元数点积
    dot_product = np.sum(pred_rot_norm * target_rot_norm, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算旋转角度误差（弧度）
    rot_error = 2 * np.arccos(np.abs(dot_product))  # (B,)
    rot_accuracy = np.mean(rot_error < rot_threshold)
    
    # 计算夹爪状态准确率
    gripper_accuracy = np.mean(np.abs(pred_gripper - target_gripper) < 0.5)
    
    metrics = {
        'traj_pos_acc_0.10': pos_accuracy,
        'traj_rot_acc_0.25': rot_accuracy,
        'traj_gripper': gripper_accuracy,
        'pos_error_mean': np.mean(pos_error),
        'rot_error_mean': np.mean(rot_error),
    }
    
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description='训练具身操控模型')
    
    parser.add_argument('--data_dir', type=str, default='./NS-2025-11-data', 
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='学习率')
    parser.add_argument('--feature_dim', type=int, default=768, 
                        help='特征维度')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='数据加载线程数')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='训练设备')
    parser.add_argument('--log_interval', type=int, default=10, 
                        help='日志记录间隔')
    parser.add_argument('--save_interval', type=int, default=5, 
                        help='模型保存间隔')
    parser.add_argument('--pos_weight', type=float, default=1.0,
                        help='位置损失权重')
    parser.add_argument('--rot_weight', type=float, default=1.0,
                        help='旋转损失权重')
    parser.add_argument('--gripper_weight', type=float, default=1.0,
                        help='夹爪状态损失权重')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='评估间隔')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='验证集比例')
    
    return parser.parse_args()

def train(model, train_loader, criterion, optimizer, epoch, device, writer, log_interval, scaler=None):
    model.train()
    epoch_loss = 0.0
    epoch_pos_loss = 0.0
    epoch_rot_loss = 0.0
    epoch_gripper_loss = 0.0
    
    all_metrics = {
        'traj_pos_acc_0.10': 0.0,
        'traj_rot_acc_0.25': 0.0,
        'traj_gripper': 0.0,
        'pos_error_mean': 0.0,
        'rot_error_mean': 0.0,
    }
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for i, batch in enumerate(pbar):
        # 获取数据并移动到GPU
        rgb = batch['rgb'].to(device, non_blocking=True)
        pcd = batch['pcd'].to(device, non_blocking=True)
        
        # 正确处理指令张量 - 这是一个形状为 (batch_size, 53, 512) 的张量
        instructions = batch['instruction'].to(device, non_blocking=True)
            
        action_target = batch['action'].to(device, non_blocking=True)
        
        # 前向传播 - 使用混合精度训练(如果可用)
        optimizer.zero_grad()
        
        if HAS_AMP and scaler is not None:
            with autocast('cuda'):
                action_pred = model(rgb, pcd, instructions)
                loss, pos_loss, rot_loss, gripper_loss = criterion(action_pred, action_target)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            action_pred = model(rgb, pcd, instructions)
            loss, pos_loss, rot_loss, gripper_loss = criterion(action_pred, action_target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
        
        # 计算评估指标
        metrics = compute_metrics(action_pred, action_target)
        
        # 更新统计信息
        epoch_loss += loss.item()
        epoch_pos_loss += pos_loss.item()
        epoch_rot_loss += rot_loss.item()
        epoch_gripper_loss += gripper_loss.item()
        
        for k, v in metrics.items():
            all_metrics[k] += v
        
        pbar.set_postfix({
            'loss': loss.item(),
            'pos_acc': metrics['traj_pos_acc_0.10'],
            'rot_acc': metrics['traj_rot_acc_0.25'],
            'grip_acc': metrics['traj_gripper']
        })
        
        # 记录损失
        if i % log_interval == 0:
            writer.add_scalar('train/batch_loss', loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('train/batch_pos_loss', pos_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('train/batch_rot_loss', rot_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('train/batch_gripper_loss', gripper_loss.item(), epoch * len(train_loader) + i)
    
    # 计算平均损失和指标
    num_batches = len(train_loader)
    epoch_loss /= num_batches
    epoch_pos_loss /= num_batches
    epoch_rot_loss /= num_batches
    epoch_gripper_loss /= num_batches
    
    for k in all_metrics:
        all_metrics[k] /= num_batches
    
    # 记录训练指标
    writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
    writer.add_scalar('train/epoch_pos_loss', epoch_pos_loss, epoch)
    writer.add_scalar('train/epoch_rot_loss', epoch_rot_loss, epoch)
    writer.add_scalar('train/epoch_gripper_loss', epoch_gripper_loss, epoch)
    
    writer.add_scalar('train/traj_pos_acc', all_metrics['traj_pos_acc_0.10'], epoch)
    writer.add_scalar('train/traj_rot_acc', all_metrics['traj_rot_acc_0.25'], epoch)
    writer.add_scalar('train/traj_gripper', all_metrics['traj_gripper'], epoch)
    
    return epoch_loss, all_metrics

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_pos_loss = 0.0
    val_rot_loss = 0.0
    val_gripper_loss = 0.0
    
    all_metrics = {
        'traj_pos_acc_0.10': 0.0,
        'traj_rot_acc_0.25': 0.0,
        'traj_gripper': 0.0,
        'pos_error_mean': 0.0,
        'rot_error_mean': 0.0,
    }
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            # 获取数据并移动到GPU
            rgb = batch['rgb'].to(device, non_blocking=True)
            pcd = batch['pcd'].to(device, non_blocking=True)
            
            # 正确处理指令张量 - 这是一个形状为 (batch_size, 53, 512) 的张量
            instructions = batch['instruction'].to(device, non_blocking=True)
                
            action_target = batch['action'].to(device, non_blocking=True)
            
            # 前向传播
            action_pred = model(rgb, pcd, instructions)
            
            # 计算损失
            loss, pos_loss, rot_loss, gripper_loss = criterion(action_pred, action_target)
            
            # 计算评估指标
            metrics = compute_metrics(action_pred, action_target)
            
            # 更新统计信息
            val_loss += loss.item()
            val_pos_loss += pos_loss.item()
            val_rot_loss += rot_loss.item()
            val_gripper_loss += gripper_loss.item()
            
            for k, v in metrics.items():
                all_metrics[k] += v
    
    # 计算平均损失和指标
    num_batches = len(val_loader)
    val_loss /= num_batches
    val_pos_loss /= num_batches
    val_rot_loss /= num_batches
    val_gripper_loss /= num_batches
    
    for k in all_metrics:
        all_metrics[k] /= num_batches
    
    return val_loss, all_metrics

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("警告: 未检测到GPU，将使用CPU进行训练，可能会非常慢！")
    
    # 设置设备
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
    
    # 任务列表
    task_names = [
        'put_money_in_safe',
        'stack_blocks',
        'light_bulb_in',
        'put_item_in_drawer',
        'place_cups'
    ]
    
    # 创建数据加载器
    # 在这里，我们将部分训练数据用作验证集
    from torch.utils.data import random_split
    from utils.data_loader import EmbodiedControlDataset
    
    full_dataset = EmbodiedControlDataset(args.data_dir, task_names, is_training=True)
    
    # 计算训练集和验证集大小
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    # 随机拆分数据集
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = MultimodalEmbodiedModel(feature_dim=args.feature_dim)
    model = model.to(device)
    
    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行数据并行训练")
        model = nn.DataParallel(model)
    
    # 定义损失函数和优化器
    criterion = CustomLoss(
        pos_weight=args.pos_weight, 
        rot_weight=args.rot_weight, 
        gripper_weight=args.gripper_weight
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 创建TensorBoard日志
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # 设置自动混合精度训练
    scaler = GradScaler() if HAS_AMP else None
    
    # 训练循环
    best_val_loss = float('inf')
    best_metrics = None
    
    print(f"开始训练，共 {args.epochs} 个epoch")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练一个轮次
        train_loss, train_metrics = train(
            model, train_loader, criterion, optimizer, epoch, device, writer, args.log_interval, scaler
        )
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Train Metrics: Pos Acc: {train_metrics['traj_pos_acc_0.10']:.4f}, "
              f"Rot Acc: {train_metrics['traj_rot_acc_0.25']:.4f}, "
              f"Gripper Acc: {train_metrics['traj_gripper']:.4f}")
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        scheduler.step()
        
        # 定期在验证集上评估
        if (epoch + 1) % args.eval_interval == 0:
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Val Metrics: Pos Acc: {val_metrics['traj_pos_acc_0.10']:.4f}, "
                  f"Rot Acc: {val_metrics['traj_rot_acc_0.25']:.4f}, "
                  f"Gripper Acc: {val_metrics['traj_gripper']:.4f}")
            
            # 记录验证指标
            writer.add_scalar('val/epoch_loss', val_loss, epoch)
            writer.add_scalar('val/traj_pos_acc', val_metrics['traj_pos_acc_0.10'], epoch)
            writer.add_scalar('val/traj_rot_acc', val_metrics['traj_rot_acc_0.25'], epoch)
            writer.add_scalar('val/traj_gripper', val_metrics['traj_gripper'], epoch)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                best_model_path = os.path.join(args.output_dir, 'best_model.pth')
                
                # 保存模型，如果使用了DataParallel，需要保存model.module
                if isinstance(model, nn.DataParallel):
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                    
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'metrics': best_metrics,
                }, best_model_path)
                
                print(f"Best model saved to {best_model_path} with val loss {best_val_loss:.6f}")
                print(f"Best Metrics: Pos Acc: {best_metrics['traj_pos_acc_0.10']:.4f}, "
                      f"Rot Acc: {best_metrics['traj_rot_acc_0.25']:.4f}, "
                      f"Gripper Acc: {best_metrics['traj_gripper']:.4f}")
        
        # 定期保存模型
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth')
            
            # 如果使用了DataParallel，需要保存model.module
            if isinstance(model, nn.DataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
                
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'metrics': train_metrics,
            }, save_path)
            
            print(f"Model saved to {save_path}")
    
    writer.close()
    print("Training completed!")
    
    if best_metrics is not None:
        print("\nBest Validation Metrics:")
        print(f"Pos Acc: {best_metrics['traj_pos_acc_0.10']:.4f}")
        print(f"Rot Acc: {best_metrics['traj_rot_acc_0.25']:.4f}")
        print(f"Gripper Acc: {best_metrics['traj_gripper']:.4f}")

if __name__ == "__main__":
    main()