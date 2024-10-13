import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from Model.Compare_Models.CrossFormer.cross_models.cross_encoder import Encoder
from Model.Compare_Models.CrossFormer.cross_models.cross_decorder import Decoder
from Model.Compare_Models.CrossFormer.cross_models.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from Model.Compare_Models.CrossFormer.cross_models.cross_embed import DSW_embedding

from math import ceil

class Crossformer(nn.Module):
    def __init__(self, data_dim, seq_length, out_len, seg_len, win_size = 2,
                factor=10, d_model=64, d_ff = 64, n_heads=8, e_layers=3, 
                dropout=0.1, baseline = False, device=torch.device('cuda:0')):
        super(Crossformer, self).__init__()
        self.data_dim = data_dim
        self.seq_length = seq_length
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_seq_length = ceil(1.0 * seq_length / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.seq_length_add = self.pad_seq_length - self.seq_length

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_seq_length // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_seq_length // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)
        
    def forward(self, x_seq):
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.seq_length_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.seq_length_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)


        return base + predict_y[:, :self.out_len, :]
    
    