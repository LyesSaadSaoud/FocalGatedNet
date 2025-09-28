import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention#, AttentionLayer#, AttentionLayer1, AttentionLayer2,  AttentionLayer_d,ProbAttention
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os

"""
# Original
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
"""
class AxialAttentionLayer(nn.Module):
    def __init__(self):
        super(AxialAttentionLayer, self).__init__()

    def forward(self, queries, keys, values):
        if queries.dim() == 3:
            # Add a dummy dimension for the head
            queries = queries.unsqueeze(2)
            keys = keys.unsqueeze(2)
            values = values.unsqueeze(2)

        B, L, H, E = queries.shape
        _, S, _, D = keys.shape

        scores_1 = torch.einsum("bhle,bshd->bhls", queries, keys)
        scores_2 = torch.einsum("bhle,bhse->bhls", queries, keys.permute(0, 2, 3, 1))
        scores = scores_1 + scores_2

        A = torch.softmax(scores, dim=-1)
        V = torch.einsum("bhls,bshd->blhd", A, values.view(B, S, H, D))

        return V.contiguous()




class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.axial_attention = AxialAttentionLayer()

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        queries = queries.view(B, L * H, E)
        keys = keys.view(B, S * H, D)
        values = values.view(B, S * H, D)

        scores = self.axial_attention(queries, keys, values)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask.unsqueeze(1).unsqueeze(2), -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values.view(B, S, H, D))

        if self.output_attention:
            return V.contiguous().view(B, L, H, E), A.contiguous().view(B, L, H, S)
        else:
            return V.contiguous().view(B, L, H, E), None

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads


        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, self.d_keys)
        keys = self.key_projection(keys).view(B, S, H, self.d_keys)
        values = self.value_projection(values).view(B, S, H, self.d_values)

        # Calculate attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn_scores = scores / (self.d_keys ** 0.5)

        # Apply attention mask
        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # Apply softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Attend to values
        attn_output = torch.einsum("bhls,bshd->blhd", attn_probs, values)
        attn_output = attn_output.view(B, L, -1)

        # Linear projection
        out = self.out_projection(attn_output)

        return out, attn_probs

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

# class Model(nn.Module):
#     """
#     Vanilla Transformer with O(L^2) complexity
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.pred_len = configs.pred_len
#         self.output_attention = configs.output_attention
#
#         # Embedding
#         if configs.embed_type == 0:
#             self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                             configs.dropout)
#             self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                            configs.dropout)
#         elif configs.embed_type == 1:
#             self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         elif configs.embed_type == 2:
#             self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#
#         elif configs.embed_type == 3:
#             self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         elif configs.embed_type == 4:
#             self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#             self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
#                                                     configs.dropout)
#         # Encoder
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                       output_attention=configs.output_attention), configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation
#                 ) for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#         # Decoder
#         self.decoder = Decoder(
#             [
#                 DecoderLayer(
#                     AttentionLayer(
#                         FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation,
#                 )
#                 for l in range(configs.d_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model),
#             projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
#         )
#
#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
#
#         enc_out = self.enc_embedding(x_enc, x_mark_enc)
#         enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
#
#         dec_out = self.dec_embedding(x_dec, x_mark_dec)
#         dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
#         # print(dec_out[:, -self.pred_len:, :].shape)
#         if self.output_attention:
#             return dec_out[:, -self.pred_len:, :], attns
#         else:
#             return dec_out[:, -self.pred_len:, :]  # [B, L, D]
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn_probs = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        out = self.out_projection(out)

        return out, attn_probs


class TemporalConvolutionBlock(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(TemporalConvolutionBlock, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, -2))))
        y = self.dropout(self.conv2(y).transpose(-1, -2))
        x = self.norm2(x + y)
        return x
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.temporal_conv_block = TemporalConvolutionBlock(d_model, d_ff, dropout, activation)

        self.feed_forward1 = nn.Linear(d_model, d_ff)
        self.feed_forward2 = nn.Linear(d_ff, d_model)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x1=x = x + self.dropout(new_x)
        x = self.temporal_conv_block(x)
        x = self.temporal_conv_block(x)
        x = self.temporal_conv_block(x)

        # Residual connection
        #x = x + self.dropout(self.feed_forward2(self.activation(self.feed_forward1(x))))

        return x, attn
# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         self.temporal_conv_block = TemporalConvolutionBlock(d_model, d_ff, dropout, activation)
#
#     def forward(self, x, attn_mask=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )
#         x = x + self.dropout(new_x)
#         x = self.temporal_conv_block(x)
#         x = self.temporal_conv_block(x)
#         x = self.temporal_conv_block(x)
#
#         # y = x = self.norm1(x)
#         #
#         # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         # y = self.dropout(self.conv2(y).transpose(-1, 1))
#
#         return x, attn #self.norm2(x + y), attn
# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         self.temporal_conv_block = TemporalConvolutionBlock(d_model, d_ff, dropout, activation)
#
#         self.feed_forward1 = nn.Linear(d_model, d_ff)
#         self.feed_forward2 = nn.Linear(d_ff, d_model)
#
#     def forward(self, x, attn_mask=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )
#         x1=x = x + self.dropout(new_x)
#         x = self.temporal_conv_block(x)
#         x = self.temporal_conv_block(x)
#         x = self.temporal_conv_block(x)
#
#         # Residual connection
#         #x = x + self.dropout(self.feed_forward2(self.activation(self.feed_forward1(x))))
#
#         return x, attn
# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#         self.temporal_conv_block = TemporalConvolutionBlock(d_model, d_ff, dropout, activation)
#
#         self.feed_forward1 = nn.Linear(d_model, d_ff)
#         self.feed_forward2 = nn.Linear(d_ff, d_model)
#
#         self.memory_attention = FullAttention()
#
#     def forward(self, x, memory=True, attn_mask=None):
#         new_x, attn = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )
#         x = x + self.dropout(new_x)
#         x = self.temporal_conv_block(x)
#         x = self.temporal_conv_block(x)
#         x = self.temporal_conv_block(x)
#
#         # Residual connection
#         x = x + self.dropout(self.feed_forward2(self.activation(self.feed_forward1(x))))
#
#         if memory is not None:
#             mem_x, mem_attn = self.memory_attention(
#                 x, memory, memory,
#                 attn_mask=attn_mask
#             )
#             x = x + self.dropout(mem_x)
#             attn = (attn, mem_attn)  # Combine self-attention and memory attention
#
#         return x, attn


# class EncoderLayer(nn.Module):
#     def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.attention = attention
#         self.temporal_conv_block = TemporalConvolutionBlock(d_model, d_ff, dropout, activation)
#
#     def forward(self, x, attn_mask=None):
#         x = self.attention(
#             x, x, x,
#             attn_mask=attn_mask
#         )
#         x = self.temporal_conv_block(x)
#         return x



# class DecoderLayer(nn.Module):
#     def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
#                  dropout=0.1, activation="relu"):
#         super(DecoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.temporal_attention1 = MultiHeadedTemporalConvolution(d_model, d_ff, dropout, activation)  # Replace conv1 and conv2
#
#         self.temporal_attention = MultiHeadedTemporalConvolution(d_model, d_ff, dropout, activation)  # Replace conv1 and conv2
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, cross, x_mask=None, cross_mask=None):
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask
#         )[0])
#         x = self.norm1(x)
#
#         x = x + self.dropout(self.cross_attention(
#             x, cross, cross,
#             attn_mask=cross_mask
#         )[0])
#
#         y = x = self.norm2(x)
#         y = self.dropout(self.temporal_attention(y))  # Use temporal_attention instead of conv1 and conv2
#
#         return self.norm3(x + y)
# class DecoderLayer(nn.Module):
#     def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
#                  dropout=0.1, activation="relu"):
#         super(DecoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.temporal_attention = MultiHeadedTemporalConvolution(d_model, d_ff, dropout,
#                                                                  activation)  # Replace conv1 and conv2
#
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, x, cross, x_mask=None, cross_mask=None):
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask
#         )[0])
#         x = self.norm1(x)
#
#         x = x + self.dropout(self.cross_attention(
#             x, cross, cross,
#             attn_mask=cross_mask
#         )[0])
#         x = self.norm1(x)
#         x = self.dropout(self.temporal_attention(x))
#         y = x = self.norm2(x)
#         y = self.dropout(self.temporal_attention(y))  # Use temporal_attention instead of conv1 and conv2
#
#         return self.norm3(x + y)
import torch
import torch.nn as nn

class MultiHeadedTemporalConvolution(nn.Module):
    def __init__(self, d_model, d_ff=None, num_heads=4, dropout=0.1, activation="relu"):
        super(MultiHeadedTemporalConvolution, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

        # Create parallel convolutional layers with different dilation rates
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, dilation=2 ** i)
            for i in range(num_heads)
        ])

        self.norm = nn.LayerNorm(4*d_model*num_heads)
        self.conv=nn.Conv1d(in_channels=4*d_model*num_heads, out_channels=d_model, kernel_size=1)

    def forward(self, x, attn_mask=None):
        # Apply parallel convolutional layers and concatenate the outputs
        conv_outputs = []
        for i in range(self.num_heads):
            conv_output = self.activation(self.conv_layers[i](x.transpose(1, 2)))
            conv_outputs.append(conv_output.transpose(1, 2))

        x = torch.cat(conv_outputs, dim=2)
        x = self.dropout(x)

        # Apply layer normalization
        #x = x.transpose(1, 2)  # Transpose to shape [batch_size, sequence_length, d_model]
        #
        x = self.conv(self.norm(x).transpose(1, 2) )#.transpose(1, 2).contiguous())  # Transpose and apply layer normalization
        x = x.transpose(1, 2)  # Transpose back to the original shape [batch_size, d_model, sequence_length]
        #print(x.shape)
        return x
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(self.activation(self.linear1(x)))
        x = self.linear2(x)
        return x

class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model, d_ff=None, num_heads=4, dropout=0.1, activation="relu"):
        super(TemporalAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

        # Linear layers for projection
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        # Linear layer for output
        self.output_projection = nn.Linear(d_model, d_model)

        # Position-wise feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        residual = x

        # Self-attention
        q = self.query_projection(x)
        k = self.key_projection(x)
        v = self.value_projection(x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(-1, attn_output.size(2), x.size(2))

        x = residual + self.dropout(self.output_projection(attn_output))

        # Feed-forward
        residual = x
        x = self.norm(x)
        x = self.feed_forward(x)
        x =  residual + self.dropout(x)

        return x
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, num_heads=4, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        #self.temporal_conv_block = TransformerXL(d_model, d_ff, num_heads, activation)
        self.temporal_attention = TemporalAttentionBlock(d_model, d_ff, dropout,
                                                                 activation)  # Replace conv1 and conv2

        self.temporal_attention1 = TemporalAttentionBlock(d_model, d_ff, dropout, activation)  # Replace conv1 and conv2
        self.temporal_attention2 = TemporalAttentionBlock(d_model, d_ff, dropout, activation)  # Replace conv1 and conv2

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.temporal_attention(y))  # Use temporal_attention instead of conv1 and conv2
        y = self.dropout(self.temporal_attention1(y))
        y = self.dropout(self.temporal_attention2(y))

        return self.norm2(x + y), attn



class Model(nn.Module):
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
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
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
        # print(dec_out[:, -self.pred_len:, :].shape)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]