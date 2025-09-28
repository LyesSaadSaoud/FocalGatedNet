import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
class Model(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.enc_in = configs.enc_in
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

        # Hierarchical Attention module
        self.channel_attn = nn.Sequential(
            nn.Linear(self.seq_len//2, self.seq_len//2 ),
            nn.ReLU(),
            nn.Linear(self.seq_len//2 , self.channels),
            nn.Softmax(dim=1)
        )
        self.temporal_attn = nn.Sequential(
            nn.Linear(42, self.seq_len ),
            nn.ReLU(),
            nn.Linear(self.seq_len, 42),
            nn.Softmax(dim=1)
        )
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
    def forward(self, x):
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Hierarchical Attention
        # Channel Attention
        channel_weights = torch.zeros(x.size(0), x.size(2), self.channels).to(x.device)
        for i in range(self.channels):

            channel_weights[:, :, i] = self.channel_attn(x[:, :, i])
        channel_weights = channel_weights.transpose(1, 2)

        # Temporal Attention
        temporal_weights = torch.zeros(x.size(0), self.seq_len//2, self.channels).to(x.device)
        for i in range(self.seq_len//2):

            temporal_weights[:, i, :] = self.temporal_attn(x[:, i, :])
        #temporal_weights = temporal_weights.transpose(1, 2)

        # Hierarchical Attention output
        attn_output = torch.bmm(channel_weights, x.transpose(1, 2))
        #print(temporal_weights.shape, attn_output.transpose(1, 2).shape)
        attn_output = torch.bmm(temporal_weights, attn_output)

        attn_output = self.Linear(attn_output.permute)
        return attn_output  # [Batch, Output length, Channel]
'''

# class Model(nn.Module):
#     """
#     Normalization-Linear
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.channels = configs.enc_in
#         self.individual = configs.individual
#         if self.individual:
#             self.Linear = nn.ModuleList()
#             for i in range(self.channels):
#                 self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
#         else:
#             self.Linear = nn.Linear(self.seq_len, self.pred_len)
#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         seq_last = x[:,-1:,:].detach()
#         x = x - seq_last
#         if self.individual:
#             output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
#             for i in range(self.channels):
#                 output[:,:,i] = self.Linear[i](x[:,:,i])
#             x = output
#         else:
#             x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
#         # print(x.shape, seq_last.shape)
#         x = x + seq_last
#
#         return x

# #LSTM
# import torch
# import torch.nn as nn
#
# class Model(nn.Module):
#     """
#     Normalization-Linear with LSTM
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.channels = configs.enc_in
#         self.individual = configs.individual
#
#         if self.individual:
#             self.LSTM = nn.LSTM(input_size=self.channels, hidden_size=self.channels, num_layers=2, batch_first=True)
#             self.Linear = nn.ModuleList()
#             for i in range(self.channels):
#                 self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
#         else:
#             self.LSTM = nn.LSTM(input_size=self.channels, hidden_size=self.channels, num_layers=2, batch_first=True)
#             self.Linear = nn.Linear(self.seq_len, self.pred_len)
#
#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         # print(x.shape)
#         x, _ = self.LSTM(x)  # Apply LSTM
#         # print(x.shape)
#         seq_last = x[:, -1:, :].detach()
#         x = x - seq_last
#
#         if self.individual:
#             output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
#             for i in range(self.channels):
#                 output[:, :, i] = self.Linear[i](x[:, :, i])
#             x = output
#         else:
#             x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
#
#         x = x + seq_last
#         # print("Model Output Shape:", x.shape)
#         return x
#GRU
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Normalization-Linear with LSTM
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.id = configs.model_id
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual

        if self.individual:
            if self.id == "LSTM":
                self.LSTM = nn.LSTM(input_size=self.channels, hidden_size=self.channels, num_layers=2, batch_first=True)

            else:

                self.LSTM = nn.GRU(input_size=self.channels, hidden_size=self.channels, num_layers=2, batch_first=True)
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            if self.id=="LSTM":

                self.LSTM = nn.LSTM(input_size=self.channels, hidden_size=self.channels, num_layers=2, batch_first=True)
            else:

                self.LSTM = nn.GRU(input_size=self.channels, hidden_size=self.channels, num_layers=2, batch_first=True)
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # print(x.shape)
        x, _ = self.LSTM(x)  # Apply LSTM
        # print(x.shape)
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = x + seq_last
        # print("Model Output Shape:", x.shape)
        return x
