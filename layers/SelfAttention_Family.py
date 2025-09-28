import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os


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
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


'''

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 top_keys=False, context_zero=False, **_):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.top_keys = top_keys
        self.context_zero = context_zero

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        if self.top_keys:
            Q_expand = Q.unsqueeze(-2).expand(B, H, L_Q, L_K, E)  # Note that expanding does not allocate new memory
            index_sample = torch.randint(L_Q, (sample_k, L_K))  # real U = U_part(factor*ln(L_k))*L_q
            Q_sample = Q_expand[:, :, index_sample, torch.arange(L_K).unsqueeze(0), :]
            Q_K_sample = torch.einsum('bhsld, bhdlr -> bhslr', Q_sample, K.unsqueeze(-3).transpose(-3, -1)).squeeze()
        else:
            K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)    # Note that expanding does not allocate new memory
            index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
            # Select sample_k number of samples from the keys
            K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
            Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        if self.top_keys:
            M = Q_K_sample.max(-2)[0] - torch.div(Q_K_sample.sum(-2), L_Q)
        else:
            M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        if self.top_keys:
            K_reduce = K[torch.arange(B)[:, None, None],
                       torch.arange(H)[None, :, None],
                       M_top, :]  # factor*ln(L_q)
            Q_K = torch.matmul(Q, K_reduce.transpose(-2, -1))  # factor*ln(L_q)*L_k
        else:
            Q_reduce = Q[torch.arange(B)[:, None, None],
                       torch.arange(H)[None, :, None],
                       M_top, :]  # factor*ln(L_q)
            Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.context_zero:
            if not self.mask_flag:
                V_sum = V.mean(dim=-2)
                contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
            else:  # use mask
                assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
                contex = V.cumsum(dim=-2)
        else:
            contex = torch.zeros(B, H, L_Q, D).to(V.device)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, L_K=L_V, top_keys=self.top_keys, device=V.device)
            scores.masked_fill_(attn_mask.mask, -1e20)      # np.inf)
            no_indxs = torch.where(torch.all(attn_mask._mask, -1))

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        if self.top_keys:
            V_sub = V[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
            context_in = torch.matmul(attn, V_sub).type_as(context_in)
            if self.mask_flag:
                context_in[no_indxs] = V[no_indxs]
        else:
            context_in[torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            if self.top_keys:
                attns = torch.zeros([B, H, L_Q, L_V]).type_as(attn).to(attn.device)
                attns[torch.arange(B)[:, None, None, None],
                      torch.arange(H)[None, :, None, None],
                      torch.arange(L_Q)[None, None, :, None],
                      index[:,:, None,:]] = attn
            else:
                attns = (torch.ones([B, H, L_Q, L_V]) / L_V).type_as(attn).to(attn.device)
                attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, *_, **__):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # Finding top keys instead of top queries
        if self.top_keys:
            u = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
            U_part = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

            U_part = U_part if U_part < L_Q else L_Q
            u = u if u < L_K else L_K
        else:
            U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
            u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

            U_part = U_part if U_part < L_K else L_K
            u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.contiguous(), attn

class NonLocalAttention(nn.Module):
    def __init__(self, d_model):
        super(NonLocalAttention, self).__init__()
        self.theta = nn.Linear(d_model, d_model)
        self.phi = nn.Linear(d_model, d_model)
        self.g = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.d_model=d_model
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H = queries.shape
        _, S, _ = keys.shape
        #print(queries.shape)
        theta = self.theta(queries).view(B, L, 1, -1)
        phi = self.phi(keys).view(B, 1, S, -1)
        g = self.g(values).view(B, S, 1, -1)
        attn_logits = torch.matmul(theta, phi.transpose(-2, -1))
        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = self.softmax(attn_logits)
        attn_weights = attn_weights.repeat(1, 4,1, 1)
        attn_output = torch.matmul(attn_weights.permute(0,3,2,1), g.permute(0,2,3,1)).view(B, L, -1)
        return attn_output, attn_weights

class RelationalAttention(nn.Module):
    def __init__(self, d_model):
        super(RelationalAttention, self).__init__()
        self.k = nn.Linear(d_model, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape

        k = self.k(keys).view(B, S, 1, -1).expand(B, S, L, -1)
        q = self.q(queries).view(B, L, 1, -1).expand(B, L, S, -1)
        v = self.v(values).view(B, S, 1, -1).expand(B, S, L, -1)
        v = v.repeat(1, 1, 1, 2)

        attn_logits = torch.matmul(torch.cat([k, q], dim=-1), v.transpose(-2, -1))
        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = self.softmax(attn_logits)

        # Use the expanded values tensor `v` instead of the original `values` tensor
        attn_output = torch.matmul(attn_weights, v).view(B, L, -1)

        return attn_output, attn_weights

class AttentionLayer1(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer1, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.non_local_attn = NonLocalAttention(d_model)
        self.relational_attn = RelationalAttention(d_model)
        self.gate_layer = nn.Linear(131712,d_model)
        self.inner_scores_correction = nn.Linear(171, d_model)
        self.non_local_scores_correction = nn.Linear(171, d_model)
        self.non_local_out_correction = nn.Linear(128, d_model)
        self.relational_scores_correction = nn.Linear(170, d_model)
        self.relational_out_correction = nn.Linear(131072, d_model)


        self.sigmoid = nn.Sigmoid()

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queriest, keyst, valuest =queries, keys, values
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply the inner attention mechanism
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        # Apply non-local attention mechanism
        non_local_out, non_local_attn = self.non_local_attn(queriest, keyst, valuest)

        # Apply relational attention mechanism
        relational_out, relational_attn = self.relational_attn(queriest, keyst, valuest)

        # Concatenate outputs of inner, non-local and relational attention
        cat_out = torch.cat([out, non_local_out, relational_out], dim=-1)

        # Compute gating scores
        gating_scores = self.gate_layer(cat_out)

        # Apply sigmoid activation to gating scores
        gating_scores = self.sigmoid(gating_scores)

        # Split gating scores into two scores for each attention mechanism
        inner_scores, non_local_scores, relational_scores = torch.chunk(gating_scores, chunks=3, dim=-1)

        # Compute weighted sum of attention outputs based on gating scores
        # print(inner_scores.shape, out.shape, non_local_scores.shape, non_local_out.shape, relational_scores.shape, relational_out.shape)
        # weighted_out = self.inner_scores_correction(inner_scores) * out + \
        #                self.non_local_scores_correction(non_local_scores) * self.non_local_out_correction(non_local_out) \
        #                + self.relational_scores_correction(relational_scores) * self.relational_out_correction(relational_out)
        weighted_out=gating_scores
        # Apply output projection layer
        out = self.out_projection(weighted_out)

        # Combine attention scores from inner, non-local and relational attention mechanisms
        attn = attn #+ non_local_attn + relational_attn

        return out, attn

class TemporalAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.out_layer = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = 1
        queries = self.q_layer(queries).view(B, L, H, -1)
        keys = self.k_layer(keys).view(B, S, H, -1)
        values = self.v_layer(values).view(B, S, H, -1)

        # Compute attention weights
        attn_weights = torch.matmul(queries, keys.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.d_model)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        out = torch.matmul(attn_weights, values).view(B, L, -1)

        # Apply output projection layer
        out = self.out_layer(out)

        # Return attention weights for visualization
        attn = attn_weights.squeeze(dim=2)
        return out, attn

class FeatureAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.out_layer = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = 1
        queries = self.q_layer(queries).view(B, L, H, -1)
        keys = self.k_layer(keys).view(B, S, H, -1)
        values = self.v_layer(values).view(B, S, H, -1)

        # Compute attention weights
        attn_weights = torch.matmul(queries, keys.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.d_model)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        out = torch.matmul(attn_weights.transpose(-1, -2), values).view(B, L, -1)

        # Apply output projection layer
        out = self.out_layer(out)

        # Return attention weights for visualization
        attn = attn_weights.squeeze(dim=2)
        return out, attn

class AttentionLayer2(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer2, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.temporal_attn = TemporalAttention(d_model)
        self.feature_attn = FeatureAttention(d_model)
        self.gate_layer = nn.Linear(3*d_model,d_model)
        self.inner_scores_correction = nn.Linear(171, d_model)
        self.temporal_scores_correction = nn.Linear(171, d_model)
        self.feature_scores_correction = nn.Linear(170, d_model)
        #self.temporal_out_correction = nn.Linear(170, d_model)
        #self.feature_out_correction = nn.Linear(512, d_model)

        self.sigmoid = nn.Sigmoid()

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queriest, keyst, valuest =queries, keys, values
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply the inner attention mechanism
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        # Apply temporal attention mechanism
        temporal_out, temporal_attn = self.temporal_attn(queriest, keyst, valuest)

        # Apply feature attention mechanism
        feature_out, feature_attn = self.feature_attn(queriest, keyst, valuest)

        # Concatenate outputs of inner, temporal and feature attention
        cat_out = torch.cat([out, temporal_out, feature_out], dim=-1)

        # Compute gating scores
        gating_scores = self.gate_layer(cat_out)

        # Apply sigmoid activation to gating scores
        #gating_scores = self.sigmoid(gating_scores)

        # Split gating scores into three scores for each attention mechanism
        inner_scores, temporal_scores, feature_scores = torch.chunk(gating_scores, chunks=3, dim=-1)
        # print(inner_scores.shape, out.shape, temporal_scores.shape, temporal_out.shape, feature_scores.shape,
        #       feature_out.shape)
        # Compute weighted sum of attention outputs based on gating scores
        weighted_out = self.inner_scores_correction(inner_scores) * out + \
                       self.temporal_scores_correction(temporal_scores) * temporal_out \
                       + self.feature_scores_correction(feature_scores)* feature_out
        # weighted_out = gating_scores
        # Apply output projection layer
        out = self.out_projection(weighted_out)

        # Combine attention scores from inner, temporal and feature attention mechanisms
        attn = attn #+ temporal_attn + feature_attn

        return out, attn

# Convolutional Attention from the LogSparse Transformer
class LogSparseAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, qk_ker, d_keys=None, d_values=None, v_conv=False, **_):
        super(LogSparseAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.qk_ker = qk_ker
        self.query_projection = nn.Conv1d(d_model, d_keys * n_heads, self.qk_ker)
        self.key_projection = nn.Conv1d(d_model, d_keys * n_heads, self.qk_ker)
        self.v_conv = v_conv
        if v_conv:
            self.value_projection = nn.Conv1d(d_model, d_values * n_heads, self.qk_ker)
        else:
            self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, **_):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = nn.functional.pad(queries.permute(0, 2, 1), pad=(self.qk_ker - 1, 0))
        queries = self.query_projection(queries).permute(0, 2, 1).view(B, L, H, -1)

        keys = nn.functional.pad(keys.permute(0, 2, 1), pad=(self.qk_ker - 1, 0))
        keys = self.key_projection(keys).permute(0, 2, 1).view(B, S, H, -1)

        if self.v_conv:
            values = nn.functional.pad(values.permute(0, 2, 1), pad=(self.qk_ker - 1, 0))
            values = self.value_projection(values).permute(0, 2, 1).view(B, S, H, -1)
        else:
            values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError('Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x

class AttentionLayer_d(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer_d, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        # self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        # self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        # self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.query_projection = LocalityFeedForward(d_model, d_keys * n_heads, 1, 4, act='hs+se', reduction=d_model // 4)
        self.key_projection = LocalityFeedForward(d_model, d_keys * n_heads, 1, 4, act='hs+se', reduction=d_model // 4)
        self.value_projection = LocalityFeedForward(d_model, d_values * n_heads, 1, 4, act='hs+se', reduction=d_model// 4)

        # self.query_projection = nn.LSTM(d_model, d_keys * n_heads, num_layers=1, bidirectional=False)
        # self.key_projection = nn.LSTM(d_model, d_keys * n_heads, num_layers=1, bidirectional=False)
        # self.value_projection = nn.LSTM(d_model, d_values * n_heads, num_layers=1, bidirectional=False)


        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        # queries, _ = self.query_projection(queries)
        # queries = queries.view(B, L, H, -1)
        # keys, _ = self.key_projection(keys)
        # keys = keys.view(B, S, H, -1)
        # values, _ = self.value_projection(values)
        # values = values.view(B, S, H, -1)


        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
'''