import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# 移除对外部库的依赖，使用自定义简化模型
HAS_TRANSFORMERS = False
HAS_POINTNET = False
print("使用自定义简化模型替代外部依赖库")

class VisualAttention(nn.Module):
    """
    视觉注意力模块，用于融合不同视角的特征
    """
    def __init__(self, feature_dim):
        super(VisualAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.register_buffer("scale", torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32)))
        
    def forward(self, x):
        """
        输入: (B, V, D) - 批次大小、视角数、特征维度
        输出: (B, D) - 批次大小、特征维度
        """
        # 计算注意力权重
        q = self.query(x)  # (B, V, D)
        k = self.key(x)    # (B, V, D)
        v = self.value(x)  # (B, V, D)
        
        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, V, V)
        attn = F.softmax(attn, dim=-1)  # (B, V, V)
        
        # 加权求和
        out = torch.matmul(attn, v)  # (B, V, D)
        
        # 对所有视角特征进行池化
        out = torch.mean(out, dim=1)  # (B, D)
        
        return out

class SimpleSwinTransformer(nn.Module):
    """
    简化版Swin Transformer，使用CNN+Transformer结构替代复杂的窗口注意力机制
    """
    def __init__(self, feature_dim=768):
        super(SimpleSwinTransformer, self).__init__()
        
        # CNN特征提取部分
        self.cnn_backbone = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第二层卷积块
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第三层卷积块
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第四层卷积块
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 全局池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 简化版Transformer编码器（用于序列建模）
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # 输出投影层
        self.proj = nn.Linear(512, feature_dim)
        
    def forward(self, x):
        # CNN特征提取
        x = self.cnn_backbone(x)  # (B, 512, H/16, W/16)
        
        # 全局池化
        x = self.avgpool(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        
        # 添加序列维度并通过Transformer处理
        x = x.unsqueeze(1)  # (B, 1, 512)
        x = self.transformer(x)  # (B, 1, 512)
        x = x.squeeze(1)  # (B, 512)
        
        # 特征投影
        x = self.proj(x)  # (B, feature_dim)
        
        return x

class RGBEncoder(nn.Module):
    """
    RGB图像编码器，使用简化版Swin Transformer
    """
    def __init__(self, pretrained=True, feature_dim=768):
        super(RGBEncoder, self).__init__()
        # 使用自定义简化Swin Transformer
        self.backbone = SimpleSwinTransformer(feature_dim=feature_dim)
        self.feature_dim = feature_dim
        
        # 添加视角间的注意力融合
        self.attention = VisualAttention(feature_dim)
        
        # 规范化和投影层
        self.norm = nn.LayerNorm(feature_dim)
        
        # 添加图像缩放层，确保输入图像符合模型要求尺寸
        self.resize = transforms.Resize((224, 224))
        
    def forward(self, x):
        """
        输入: (B, 4, 3, 256, 256) - 4个视角的RGB图像
        输出: (B, feature_dim) - 融合后的视觉特征
        """
        B, V, C, H, W = x.shape
        x = x.view(B * V, C, H, W)
        
        # 调整图像大小为224x224以匹配Swin Transformer的要求
        x = self.resize(x)
        
        # 提取特征
        features = self.backbone(x)  # (B*V, feature_dim)
        
        # 重塑为 (B, V, feature_dim)
        features = features.view(B, V, -1)
        
        # 使用注意力机制融合不同视角
        features = self.attention(features)  # (B, feature_dim)
        features = self.norm(features)
        
        return features

class SimplePointNetPlusPlus(nn.Module):
    """
    简化版PointNet++，使用CNN处理点云数据
    """
    def __init__(self, feature_dim=768):
        super(SimplePointNetPlusPlus, self).__init__()
        self.feature_dim = feature_dim
        
        # 使用类似ResNet的卷积网络结构处理点云
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第四个卷积块
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征映射层
        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        """
        输入: (B, 3, H, W) - 批次大小，通道，高度，宽度
        输出: (B, feature_dim) - 批次大小，特征维度
        """
        # 卷积处理
        x = self.conv_layers(x)
        
        # 全局池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 特征映射
        features = self.fc(x)
        
        return features

class PointCloudEncoder(nn.Module):
    """
    点云编码器，使用简化版PointNet++
    """
    def __init__(self, feature_dim=768):
        super(PointCloudEncoder, self).__init__()
        self.feature_dim = feature_dim
        
        # 使用自定义简化版PointNet++
        self.backbone = SimplePointNetPlusPlus(feature_dim=feature_dim)
        
        # 视角融合的注意力机制
        self.attention = VisualAttention(feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        
        # 添加点云处理的预处理 - 调整大小
        self.resize = transforms.Resize((224, 224))
        
    def forward(self, x):
        """
        输入: (B, 4, 3, 256, 256) - 批次大小，视角数，通道，高度，宽度
        输出: (B, feature_dim) - 批次大小，特征维度
        """
        B, V, C, H, W = x.shape
        x = x.reshape(B * V, C, H, W)
        
        # 调整点云大小
        x = self.resize(x)
        
        # 使用简化版PointNet++处理
        features = self.backbone(x)  # (B*V, feature_dim)
        
        # 重塑为 (B, V, feature_dim)
        features = features.reshape(B, V, -1)
        
        # 使用注意力机制融合不同视角
        features = self.attention(features)
        features = self.norm(features)
        
        return features

class SimpleCLIPEncoder(nn.Module):
    """
    简化版CLIP文本编码器，使用Transformer结构
    """
    def __init__(self, vocab_size=30522, max_length=77, feature_dim=768):
        super(SimpleCLIPEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.feature_dim = feature_dim
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, feature_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, feature_dim))
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=feature_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=6
        )
        
        # 规范化层
        self.ln_final = nn.LayerNorm(feature_dim)
        
        # 文本投影层
        self.text_projection = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, input_ids, attention_mask=None):
        """
        输入:
        - input_ids: (B, L) - 批次大小，序列长度
        - attention_mask: (B, L) - 批次大小，序列长度
        输出: (B, feature_dim) - 批次大小，特征维度
        """
        B, L = input_ids.shape
        L = min(L, self.max_length)
        
        # 如果没有注意力掩码，创建一个
        if attention_mask is None:
            attention_mask = torch.ones((B, L), device=input_ids.device)
        
        # 获取词嵌入
        token_embeddings = self.token_embedding(input_ids[:, :L])  # (B, L, D)
        
        # 添加位置编码
        embeddings = token_embeddings + self.position_embedding[:, :L, :]  # (B, L, D)
        
        # 创建注意力掩码
        mask = torch.zeros((B, L), device=input_ids.device, dtype=torch.bool)
        mask = mask.masked_fill(attention_mask[:, :L] == 0, True)  # 用于transformer的掩码
        
        # 通过Transformer编码器
        encoded = self.transformer(embeddings)  # (B, L, D)
        
        # 使用[CLS]标记或平均池化获取序列表示
        # 这里我们使用第一个标记（类似于BERT的[CLS]）
        pooled_output = encoded[:, 0]  # (B, D)
        
        # 规范化并投影
        pooled_output = self.ln_final(pooled_output)
        text_features = self.text_projection(pooled_output)  # (B, D)
        
        return text_features

class TextEncoder(nn.Module):
    """
    文本编码器，使用简化版CLIP文本编码器
    """
    def __init__(self, feature_dim=768):
        super(TextEncoder, self).__init__()
        self.feature_dim = feature_dim
        
        # 使用自定义简化版CLIP文本编码器
        self.backbone = SimpleCLIPEncoder(feature_dim=feature_dim)
        
        # 文本预处理 - 简单分词器
        self.max_length = 77
        
    def preprocess(self, text_list):
        """
        简化的分词处理，仅用于演示
        实际应用中应使用更复杂的分词器
        """
        # 创建一个简单的词汇映射
        unique_words = set()
        for text in text_list:
            unique_words.update(text.lower().split())
        word_to_id = {word: i+2 for i, word in enumerate(sorted(unique_words))}
        word_to_id["[PAD]"] = 0
        word_to_id["[UNK]"] = 1
        
        # 分词并转换为ID
        input_ids = []
        attention_masks = []
        
        for text in text_list:
            tokens = text.lower().split()[:self.max_length-2]  # 限制长度
            ids = [word_to_id.get(word, 1) for word in tokens]  # 获取ID，未知词为[UNK](1)
            
            # 填充序列
            if len(ids) < self.max_length:
                attention_mask = [1] * len(ids) + [0] * (self.max_length - len(ids))
                ids = ids + [0] * (self.max_length - len(ids))
            else:
                attention_mask = [1] * self.max_length
                ids = ids[:self.max_length]
            
            input_ids.append(ids)
            attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks)
        }
        
    def forward(self, text_list):
        """
        输入: list[str] - 文本指令列表
        输出: (B, feature_dim) - 批次大小，特征维度
        """
        # 文本预处理
        text_inputs = self.preprocess(text_list)
        
        # 使用CLIP编码器处理
        text_features = self.backbone(
            text_inputs["input_ids"], 
            text_inputs["attention_mask"]
        )
        
        return text_features

class CrossModalFusion(nn.Module):
    """
    跨模态特征融合模块，使用自注意力融合视觉和语言特征
    """
    def __init__(self, feature_dim=768):
        super(CrossModalFusion, self).__init__()
        self.feature_dim = feature_dim
        
        # 使用简化版Transformer进行跨模态注意力
        self.cross_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=feature_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # 特征融合映射层
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        
    def forward(self, rgb_features, pc_features, text_features):
        """
        输入:
        - rgb_features: (B, feature_dim) - RGB特征
        - pc_features: (B, feature_dim) - 点云特征
        - text_features: (B, feature_dim) - 文本特征
        输出:
        - fused_features: (B, feature_dim) - 融合特征
        """
        B = rgb_features.shape[0]
        
        # 创建序列：[rgb_features, pc_features, text_features]
        sequence = torch.stack([rgb_features, pc_features, text_features], dim=1)  # (B, 3, feature_dim)
        
        # 交叉注意力融合
        fused_sequence = self.cross_attention(sequence)  # (B, 3, feature_dim)
        
        # 平坦化所有特征并连接
        fused_features = fused_sequence.reshape(B, -1)  # (B, 3*feature_dim)
        
        # 映射到统一特征空间
        fused_features = self.fusion_layer(fused_features)  # (B, feature_dim)
        
        return fused_features

class ActionPrediction(nn.Module):
    """
    动作预测模块，预测机器人动作序列
    """
    def __init__(self, feature_dim=768, action_dim=6, max_seq_len=20):
        super(ActionPrediction, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len
        
        # 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # LSTM用于序列生成
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 动作预测层
        self.action_head = nn.Linear(feature_dim, action_dim)
        
    def forward(self, features, target_actions=None, teacher_forcing_ratio=0.5):
        """
        输入:
        - features: (B, feature_dim) - 融合后的多模态特征
        - target_actions: (B, seq_len, action_dim) - 目标动作序列（仅训练时需要）
        - teacher_forcing_ratio: float - 使用教师强制的概率（仅训练时有效）
        输出:
        - predicted_actions: (B, max_seq_len, action_dim) - 预测的动作序列
        """
        batch_size = features.shape[0]
        
        # 特征转换
        transformed_features = self.feature_transform(features)  # (B, feature_dim)
        
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(2, batch_size, self.feature_dim, device=features.device)
        c0 = torch.zeros(2, batch_size, self.feature_dim, device=features.device)
        
        # 初始化第一个输入（使用变换后的特征）
        decoder_input = transformed_features.unsqueeze(1)  # (B, 1, feature_dim)
        
        # 存储所有预测
        all_predictions = []
        
        # 自回归生成动作序列
        for t in range(self.max_seq_len):
            # LSTM前向传播 - 输出: (B, 1, feature_dim), 隐藏状态: (2, B, feature_dim)
            output, (h0, c0) = self.lstm(decoder_input, (h0, c0))
            
            # 预测当前时间步的动作
            current_pred = self.action_head(output.squeeze(1))  # (B, action_dim)
            
            # 保存预测
            all_predictions.append(current_pred)
            
            # 下一个时间步的输入
            if target_actions is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # 教师强制：使用真实动作作为下一输入（仅在训练时）
                if t < target_actions.shape[1] - 1:
                    next_action = target_actions[:, t, :]
                else:
                    # 如果已经超过了目标动作序列长度，则使用自己的预测
                    next_action = current_pred
            else:
                # 使用自己的预测作为下一输入
                next_action = current_pred
            
            # 将动作投影回特征空间作为下一个输入
            decoder_input = self.feature_transform(next_action)  # (B, feature_dim)
            decoder_input = decoder_input.unsqueeze(1)  # (B, 1, feature_dim)
        
        # 将所有预测堆叠成一个序列
        predicted_actions = torch.stack(all_predictions, dim=1)  # (B, max_seq_len, action_dim)
        
        return predicted_actions

class MultimodalEmbodiedModel(nn.Module):
    """
    多模态机器人控制模型，结合视觉、点云和语言指令预测动作
    """
    def __init__(self, feature_dim=768, action_dim=8):
        super(MultimodalEmbodiedModel, self).__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        # 模态编码器
        self.rgb_encoder = RGBEncoder(feature_dim=feature_dim)
        self.pc_encoder = PointCloudEncoder(feature_dim=feature_dim)
        
        # 修改文本编码器以处理张量格式的指令
        self.text_projection = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 跨模态融合
        self.fusion_module = CrossModalFusion(feature_dim=feature_dim)
        
        # 动作预测 - 简化为直接预测单个动作向量
        self.action_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, action_dim)
        )
        
    def forward(self, rgb_images, point_clouds, instructions):
        """
        输入:
        - rgb_images: (B, 4, 3, H, W) - 批次大小、视角数、通道、高度、宽度
        - point_clouds: (B, 4, 3, H, W) - 批次大小、视角数、通道、高度、宽度
        - instructions: (B, 53, 512) - 批次大小、序列长度、特征维度
        输出:
        - predicted_actions: (B, 1, action_dim) - 预测的动作向量
        """
        # 编码视觉模态
        rgb_features = self.rgb_encoder(rgb_images)  # (B, feature_dim)
        pc_features = self.pc_encoder(point_clouds)  # (B, feature_dim)
        
        # 处理文本指令张量 - 输入是(B, 53, 512)
        # 先对序列维度进行平均池化，得到(B, 512)
        text_features = torch.mean(instructions, dim=1)  # (B, 512)
        # 然后投影到所需的特征维度
        text_features = self.text_projection(text_features)  # (B, feature_dim)
        
        # 跨模态融合
        fused_features = self.fusion_module(rgb_features, pc_features, text_features)  # (B, feature_dim)
        
        # 预测动作 - 简化为单个动作向量
        action_pred = self.action_predictor(fused_features)  # (B, action_dim)
        action_pred = action_pred.unsqueeze(1)  # 添加序列维度 (B, 1, action_dim)
            
        return action_pred
    
    def predict(self, rgb_images, point_clouds, instructions):
        """
        用于推理的便捷方法
        """
        self.eval()
        with torch.no_grad():
            return self.forward(rgb_images, point_clouds, instructions)