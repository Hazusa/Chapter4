import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ----------------------
# 公共组件定义
# ----------------------
class PositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_embed[:, :x.size(1), :]


class InformerLiteBlock(nn.Module):
    def __init__(self, d_model, num_heads, factor):
        super().__init__()
        self.attention = ProbAttention(d_model, num_heads, factor)  # 使用稀疏注意力
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 稀疏注意力
        attn_out = self.attention(x)  # (B, T, d_model)

        # 降采样卷积（缩减序列长度）
        x_conv = x.permute(0, 2, 1)  # (B, d_model, T)
        x_conv = self.conv(x_conv)  # (B, d_model, T//2)
        x_conv = x_conv.permute(0, 2, 1)  # (B, T//2, d_model)

        # 残差连接 + 层归一化
        x = self.norm(attn_out[:, ::2, :] + x_conv)  # 注意维度对齐
        return self.dropout(x)
# ----------------------
# Informer-Lite 组件
# ----------------------
class ProbAttention(nn.Module):
    """稀疏概率注意力机制"""

    def __init__(self, d_model: int, num_heads: int, factor: int = 5):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.factor = factor
        self.qkv = nn.Linear(d_model, 3 * d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [y.view(B, T, self.num_heads, self.d_k).transpose(1, 2) for y in qkv]

        # 稀疏采样
        U = self.factor * int(math.log(T))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        top_k = torch.topk(scores, U, dim=-1, largest=True)[0]

        attn = F.softmax(top_k, dim=-1)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).reshape(B, T, -1)


# ----------------------
# Autoformer-Tiny 组件
# ----------------------
class SeriesDecomposition(nn.Module):
    """序列分解模块"""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> tuple:
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    """自相关注意力机制"""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        q = self.query(x)
        k = self.key(x)
        scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        return torch.matmul(scores, x)


# ----------------------
# LiteTST 组件
# ----------------------
class ChannelIndependentAttention(nn.Module):
    """通道独立注意力机制"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.size()
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [y.view(B, T, self.num_heads, self.d_k).transpose(1, 2) for y in qkv]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).reshape(B, T, -1)


# ----------------------
# 完整模型定义
# ----------------------
class InformerLite(nn.Module):
    def __init__(self, input_dim: 15, num_classes: 4, d_model=64, num_heads=4, factor=5):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=256) for _ in range(2)]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = self.pos_enc(x)
        x = x.permute(1, 0, 2)  # (T, B, d_model)
        x = self.encoder(x)
        x = x.mean(dim=0)  # (B, d_model)
        return self.classifier(x)


class AutoformerTiny(nn.Module):
    def __init__(self, input_dim=15, num_classes=4, d_model=64, moving_avg=25):
        super().__init__()
        self.decomp = SeriesDecomposition(moving_avg)
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)  # 新增位置编码
        self.trend_proj = nn.Linear(input_dim, d_model)  # 新增趋势项投影

        # 增强编码器
        self.encoder = nn.Sequential(
            *[nn.TransformerEncoderLayer(
                d_model, 4,
                dim_feedforward=128,
                dropout=0.1  # 添加dropout
            ) for _ in range(3)]  # 增加层数
        )

        # 增强分类头
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model),  # 融合季节和趋势信息
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        seasonal, trend = self.decomp(x)

        # 分别处理季节和趋势项
        seasonal = self.embed(seasonal)
        seasonal = self.pos_enc(seasonal)
        trend_proj = self.trend_proj(trend)

        # 通过编码器
        enc_out = self.encoder(seasonal.permute(1, 0, 2))

        # 信息融合
        enc_out = enc_out.mean(dim=0)  # (B, d_model)
        trend_feat = trend_proj.mean(dim=1)  # (B, d_model)

        return self.classifier(torch.cat([enc_out, trend_feat], dim=1))
"""
class AutoformerTiny(nn.Module):
    def __init__(self, input_dim: 15, num_classes: 4, d_model=64, moving_avg=25):
        super().__init__()
        self.decomp = SeriesDecomposition(moving_avg)
        self.embed = nn.Linear(input_dim, d_model)
        self.encoder = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model, 4, dim_feedforward=128) for _ in range(2)]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        seasonal, trend = self.decomp(x)
        x = self.embed(seasonal)
        x = x.permute(1, 0, 2)  # (T, B, d_model)
        x = self.encoder(x)
        x = x.mean(dim=0)  # (B, d_model)
        return self.classifier(x)

"""

class LiteTST(nn.Module):
    def __init__(self, input_dim: 15, num_classes: 4, d_model=64, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=128) for _ in range(3)]
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = x.permute(1, 0, 2)  # (T, B, d_model)
        x = self.blocks(x)
        x = x.mean(dim=0)  # (B, d_model)
        return self.classifier(x)






# =====================
# CNN模型（1D卷积）
# =====================
class TS_CNN(nn.Module):
    def __init__(self, input_dim=15, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),  # 输入通道数=input_dim
            nn.ReLU(),
            nn.MaxPool1d(2, stride=1),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局池化
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x形状: (B, T, 15) → 转换为(B, 15, T)适应Conv1d
        x = x.permute(0, 2, 1)
        x = self.features(x)  # (B, 128, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# =====================
# LSTM模型
# =====================
class TS_LSTM(nn.Module):
    def __init__(self, input_dim=15, num_classes=4, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x形状: (B, T, 15)
        output, (h_n, _) = self.lstm(x)
        # 取最后一个时间步输出
        return self.classifier(output[:, -1, :])

# =====================
# 双向LSTM模型
# =====================
class TS_BiLSTM(nn.Module):
    def __init__(self, input_dim=15, num_classes=4, hidden_size=64):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(2 * hidden_size, num_classes)  # 双向拼接

    def forward(self, x):
        output, (h_n, _) = self.bilstm(x)
        return self.classifier(output[:, -1, :])


class TS_BERT(nn.Module):
    def __init__(self, input_dim=15, num_classes=4, hidden_size=64):
        super().__init__()
        # 添加位置编码层
        self.pos_encoder = PositionalEncoding(hidden_size)

        # 修改embed初始化方式
        self.embed = nn.Linear(input_dim, hidden_size)
        nn.init.xavier_normal_(self.embed.weight)

        # 增强编码器结构
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1  # 添加dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.encoder_norm = nn.LayerNorm(hidden_size)  # 添加层归一化

        # 增强分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # 初始化CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)  # 缩小初始化范围
        self.proj = nn.Linear(input_dim, hidden_size) if input_dim != hidden_size else nn.Identity()

    def forward(self, x):
        # 修改残差连接实现
        identity = self.proj(x)  # 维度匹配投影
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = self.embed(x) + identity  # 现在维度一致

        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoder(x)  # 添加位置编码
        x = x.permute(1, 0, 2)

        # 通过编码器
        x = self.encoder(x)
        x = self.encoder_norm(x)  # 层归一化

        # 使用所有token的加权平均
        cls_output = x[0] + 0.5 * x.mean(dim=0)  # 结合全局信息
        return self.classifier(cls_output)

"""# =====================
# 简化版BERT（适应时间序列）
# =====================
class TS_BERT(nn.Module):
    def __init__(self, input_dim=15, num_classes=4, hidden_size=64):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=256,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))  # 类似[CLS]标记
        self.classifier = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        # 添加可学习的[CLS]标记
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = self.embed(x)  # (B, T, 15) → (B, T, hidden_size)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, hidden_size)
        x = x.permute(1, 0, 2)  # (T+1, B, hidden_size)
        x = self.encoder(x)  # (T+1, B, hidden_size)
        # 取CLS标记输出
        return self.classifier(x[0])

        # =====================
"""
# 简化版RoBERTa（动态掩码）
# =====================
"""class TS_RoBERTa(TS_BERT):
    def __init__(self, input_dim=15, num_classes=4, hidden_size=64):
        super().__init__(input_dim, num_classes, hidden_size)
        # 添加动态掩码机制
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, mask_ratio=0.15):
        # 生成随机掩码
        B, T, _ = x.shape
        num_mask = int(T * mask_ratio)
        rand_indices = torch.rand(B, T).argsort(dim=1)
        mask_pos = rand_indices[:, :num_mask]

        # 嵌入与掩码替换
        x_embed = self.embed(x)
        mask_embed = self.mask_embed.expand(B, num_mask, -1)
        x_embed.scatter_(1, mask_pos.unsqueeze(-1).expand(-1, -1, x_embed.size(-1)), mask_embed)

        # 后续处理与TS_BERT相同
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x_embed], dim=1)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        return self.classifier(x[0])"""


class TS_RoBERTa(TS_BERT):
    def __init__(self, input_dim=15, num_classes=4, hidden_size=64):
        super().__init__(input_dim, num_classes, hidden_size)
        # 增强动态掩码机制
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.mask_embed, std=0.02)  # 缩小初始化范围

        # 添加掩码层归一化
        self.mask_norm = nn.LayerNorm(hidden_size)

        # 增强分类头（继承父类已修改）

    def forward(self, x, mask_ratio=0.15):
        # 生成随机掩码（保持原有逻辑）
        B, T, _ = x.shape
        num_mask = int(T * mask_ratio)
        rand_indices = torch.rand(B, T).argsort(dim=1)
        mask_pos = rand_indices[:, :num_mask]

        # 使用父类的embed和残差连接
        identity = self.proj(x)
        x_embed = self.embed(x) + identity

        # 应用动态掩码
        mask_embed = self.mask_embed.expand(B, num_mask, -1)
        x_embed.scatter_(1, mask_pos.unsqueeze(-1).expand(-1, -1, x_embed.size(-1)), mask_embed)
        x_embed = self.mask_norm(x_embed)  # 添加层归一化

        # 继承父类的后续处理
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x_embed], dim=1)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = self.encoder_norm(x)

        # 使用改进后的分类方式
        cls_output = x[0] + 0.5 * x.mean(dim=0)
        return self.classifier(cls_output)


