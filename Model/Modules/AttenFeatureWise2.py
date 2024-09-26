"""
    In this module, it is kind of similar to the AttenFeatureWise.py.
    The difference is, in this module, it directly uses the MultiheadAttention module from torch.nn.
    And hence, got no dimentionality bugs.

    The AttenFeatureWisePI.py is using this module for solution_u part for predicting better initial ypred.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAttention(nn.Module):
    def __init__(self, feature_dim, d_model=64, n_heads=8, d_ff=256, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.d_model = d_model
        
        # 输入层，将每个特征维度提升到 d_model 维度
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        
        # 前馈神经网络层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, 1)  # 最终输出为一个预测值
        )
        
        # 注意力权重的层
        self.attn_weights = None
        
    def forward(self, x):
        # 输入数据形状 [batch_size, 49, 1] -> [batch_size, 49, d_model]
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        
        # 转换形状以适应注意力机制 [batch_size, 49, d_model] -> [49, batch_size, d_model]
        x = x.permute(1, 0, 2)
        
        # 计算自注意力，输出和注意力权重
        attn_output, attn_weights = self.attention(x, x, x)
        
        # 存储注意力权重以便分析
        self.attn_weights = attn_weights
        
        # 转换回原来的形状 [49, batch_size, d_model] -> [batch_size, 49, d_model]
        attn_output = attn_output.permute(1, 0, 2)
        
        # 将注意力加权后的输出通过前馈神经网络
        output = self.ffn(attn_output.mean(dim=1))  # 在时间步上取平均
        
        return output

# 使用这个模块来预测SOH
class SOHPredictor(nn.Module):
    def __init__(self, args, feature_dim=1, d_model=64, n_heads=8, d_ff=256, dropout=0.1):
        super(SOHPredictor, self).__init__()
        self.feature_attention = FeatureAttention(feature_dim, d_model, n_heads, d_ff, dropout)

    def forward(self, x):
        # 输入数据形状为 [batch_size, 49, 1]
        output = self.feature_attention(x)
        return output

    def get_attention_weights(self):
        # 返回注意力权重以便分析哪些特征最重要
        return self.feature_attention.attn_weights

