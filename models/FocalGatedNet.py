#
# Copyright Lyes SAAD SAOUD 2023
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.FocalGatedNet_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class NovelSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(NovelSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale_factor = self.head_dim ** -0.5

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        bsz, seq_len, _ = x.size()

        # Compute queries, keys, and values
        q = self.q(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute dot product attention
        dot_product = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor

        # Apply mask if provided
        if mask is not None:
            dot_product = dot_product.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attention_weights = F.softmax(dot_product, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        weighted_sum = torch.matmul(attention_weights, v)

        # Concatenate heads and apply output projection
        concatenated = weighted_sum.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        output = self.out(concatenated)

        return output


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [

                DecoderLayer(
                    NovelSelfAttention(configs.d_model, configs.n_heads, dropout=configs.dropout),
                    NovelSelfAttention(configs.d_model, configs.n_heads, dropout=configs.dropout),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
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
