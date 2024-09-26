"""
    Almost the same as the AttenFeatureWise.py, the differences are as follows:
    1. the input dimension is 49, the seq_length is 1 so that means we are treating the input as a sequence of 49 features
    2. this model is used for exploring the relationship between the xt-1 xt-2 ... xt-k and yt-1 yt-2 ... yt-k which is the temporal relationship
    3. the model is using the max pooling method in the very end of the forward function to only output xt and yt. 

    Also, this is not a complete model. It is just a module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'




class MICPearsonAttention(nn.Module):
    def __init__(self, seq_length, d_model, n_heads, d_key, d_value, dropout=0.1):
        super(MICPearsonAttention, self).__init__()
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.query = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.key = nn.Linear(d_model, n_heads* d_key, bias=False)
        self.value = nn.Linear(d_model, n_heads * d_value, bias=False)
        self.fc = nn.Linear(n_heads * d_value, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def scaled_dot_product_attention(self, Q, K, V):
        raw_weights = torch.matmul(Q, K.transpose(-2, -1)) 
        scale_factor = K.size(-1) ** 0.5
        scaled_weights = raw_weights / scale_factor    
        # Apply softmax to the combined scores
        attention_weights = F.softmax(scaled_weights, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query, key, value):
        batch_size = query.size(0)
        """
        print("batch size: ", batch_size)
        print("query shape: ", query.shape)
        print("key shape: ", key.shape)
        print("value shape: ", value.shape)
        """
        residual = query
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        """
        print("Q shape: ", Q.shape)
        print("K shape: ", K.shape)
        print("V shape: ", V.shape)
        """
        Q = Q.view(batch_size, self.seq_length, self.n_heads, self.d_key).transpose(1, 2)
        K = K.view(batch_size, self.seq_length, self.n_heads, self.d_key).transpose(1, 2)
        V = V.view(batch_size, self.seq_length, self.n_heads, self.d_value).transpose(1, 2)
        """
        print("Q shape after transpose: ", Q.shape)
        print("K shape after transpose: ", K.shape)
        print("V shape after transpose: ", V.shape)
        """
        outputs, attention_weights = self.scaled_dot_product_attention(Q, K, V)
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, self.seq_length, self.d_model)
        outputs = self.dropout(self.fc(outputs))
        outputs = self.layer_norm(residual + outputs)
        return outputs 

class EncoderLayer(nn.Module):
    def __init__(self, seq_length, d_model, n_heads, d_ff, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MICPearsonAttention(seq_length, d_model, n_heads, d_k, d_v, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, seq_length, d_model, n_heads, n_layers, d_ff, d_k, d_v, dropout=0.1):
        super(Encoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(seq_length, d_model, n_heads, d_ff, d_k, d_v, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)
    
class MultiHeadAtten(nn.Module):
    def __init__(self, args, input_dim=49, d_model=64, n_heads=8, n_layers=4, d_ff=256, d_k=8, d_v=8, dropout=0.1, seq_length = 1):
        super(MultiHeadAtten, self).__init__()

        self.encoder = Encoder(input_dim, seq_length, d_model, n_heads, n_layers, d_ff, d_k, d_v, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            Sin(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        encoded_x = self.encoder(x)
        pooled, _ = encoded_x.max(dim=1)  # max returns both values and indices, so we use _
        output = self.decoder(pooled)
        return output