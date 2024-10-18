import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from einops import rearrange, repeat



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

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_length, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        # 可学习的位置嵌入
        self.position_embeddings = nn.Embedding(seq_length, d_model)
        
    def forward(self, x):
        # x 的形状为 [batch_size, seq_length, d_model]
        batch_size, seq_length, d_model = x.size()
        # 创建位置索引 [0, 1, 2, ..., seq_length - 1]
        position_indices = torch.arange(seq_length, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        
        # 获取位置编码，并加到输入的x上
        position_encoded = self.position_embeddings(position_indices)
        return x + position_encoded

class Encoder(nn.Module):
    def __init__(self, input_dim, seq_length, d_model, n_heads, n_layers, d_ff, d_k, d_v, dropout=0.1):
        super(Encoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enbed = LearnablePositionalEncoding(seq_length, d_model)
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

class DecoderLayer(nn.Module):
    '''
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    '''
    def __init__(self, seg_len, d_model, d_ff=None, dropout=0.1, out_seg_num = 10, factor = 10):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, out_seg_num)

    def forward(self, x):
        '''
        x: the output of last decoder layer
        '''
        batch = x.shape[0]
        x = x + self.dropout(x)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x+y)
        layer_predict = self.linear_pred(dec_output)

        return dec_output, layer_predict

class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_model, d_ff, dropout,\
                router=False, out_seg_num = 10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(DecoderLayer(seg_len, d_model, d_ff, dropout, \
                                        out_seg_num, factor))

    def forward(self, x):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            x, layer_predict = layer(x)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)

        return final_predict


class AttenSpatioEmbedding(nn.Module):
    def __init__(self, args, input_dim=49, d_model=64, n_heads=8, n_layers=4, d_ff=256, d_k=8, d_v=8, dropout=0.1, seq_length = 1):
        super(AttenSpatioEmbedding, self).__init__()

        self.encoder = Encoder(input_dim, seq_length, d_model, n_heads, n_layers, d_ff, d_k, d_v, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32,60),
            nn.ReLU(),
            nn.Linear(60,32),
            nn.ReLU(),
            nn.Linear(32,seq_length)
        )
        self.Decoder = Decoder(seq_length, n_layers, d_model, d_ff, dropout, out_seg_num = seq_length)

    def forward(self, x):
        encoded_x = self.encoder(x)
        output = self.Decoder(encoded_x)
        pred_y = output[:,:,-1]
        return pred_y