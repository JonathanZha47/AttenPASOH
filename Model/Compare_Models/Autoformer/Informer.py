import torch
import torch.nn as nn
from Model.Compare_Models.Autoformer.AutoformerLayer.SelfAttention_Family import ProbAttention, AttentionLayer
from Model.Compare_Models.Autoformer.AutoformerLayer.Embed import DataEmbedding
from Model.Compare_Models.Autoformer.AutoformerLayer.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, ConvLayer


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, args):
        super(Informer, self).__init__()
        self.pred_len = args.pred_len
        self.output_attention = args.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(args.enc_in, args.d_model, args.embed, args.freq,
                                           args.dropout)
        self.dec_embedding = DataEmbedding(args.dec_in, args.d_model, args.embed, args.freq,
                                           args.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=args.output_attention),
                        args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            [
                ConvLayer(
                    args.d_model
                ) for l in range(args.e_layers - 1)
            ] if args.distil else None,
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads),
                    AttentionLayer(
                        ProbAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for l in range(args.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model),
            projection=nn.Linear(args.d_model, args.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
