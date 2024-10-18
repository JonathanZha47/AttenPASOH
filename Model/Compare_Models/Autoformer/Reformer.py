import torch
import torch.nn as nn
from Model.Compare_Models.Autoformer.AutoformerLayer.SelfAttention_Family import ReformerLayer
from Model.Compare_Models.Autoformer.AutoformerLayer.Embed import DataEmbedding
from Model.Compare_Models.Autoformer.AutoformerLayer.Autoformer_EncDec import Encoder, EncoderLayer


class Reformer(nn.Module):
    """
    Reformer with O(LlogL) complexity
    - It is notable that Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch) can be very helpful, if you have any questiones.
    """

    def __init__(self, args):
        super(Reformer, self).__init__()
        self.pred_len = args.pred_len
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq,
                                           args.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, args.d_model, args.n_heads, bucket_size=args.bucket_size,
                                  n_hashes=args.n_hashes),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        self.projection = nn.Linear(args.d_model, args.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # Reformer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
        else:
            return enc_out[:, -self.pred_len:, :]  # [B, L, D]
