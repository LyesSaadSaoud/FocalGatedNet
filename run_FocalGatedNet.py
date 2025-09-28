import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=0, help='status') #was 1
parser.add_argument('--model_id', type=str, required=False, default='NewModelx2', help='model id')
parser.add_argument('--model', type=str, required=False, default='TempoFormer',
                    help='model name, options: [EDLstm, TempoFormerModified, TempoFormer, TempoNet, Autoformer, Informer, Transformer , FocalGatedNet, PatchTST]')

# data loader
parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='selected_features_advanced.csv', help='data file')   # selected_features_advanced for my model 1% rank
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='knee_sagittal', help='target feature in S or MS task') #for EMG is LTA_norm
parser.add_argument('--freq', type=str, default='t',
                    help='freq for time features encoding, options:[us:microseconds, s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=128, help='input sequence length') #128
parser.add_argument('--label_len', type=int, default=64, help='start token length') #48
parser.add_argument('--pred_len', type=int, default=100, help='prediction sequence length') #64

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
dim=23
# nonseq2seq comon
parser.add_argument('--input_size', type=int, default=dim, help='input features dim')
## LSTNet
parser.add_argument('--hidRNN', default=100, help='RNN hidden zie')
parser.add_argument('--hidCNN', type=int, default=100, help='CNN hidden size')
parser.add_argument('--hidSkip', type=float, default=5)
parser.add_argument('--skip', type=int, default=20)
parser.add_argument('--CNN_kernel', default=6, help='kernel size')
parser.add_argument('--highway_window', type=float, default=20, help='ar regression used last * items')

## tcn
parser.add_argument('--tcn_n_layers', default=3, help='num_layers')
parser.add_argument('--tcn_hidden_size', type=int, default=64, help='tcn hidden size')
parser.add_argument('--tcn_dropout', type=float, default=0.05, help='dropout')




parser.add_argument('--rnn_hidden_size', type=int, default=64, help='CNN hidden size')
parser.add_argument('--rnn_n_layers', type=int, default=3, help='CNN hidden size')
parser.add_argument('--rnn_dropout', type=float, default=0.05, help='dropout')
# model common
parser.add_argument('--out_size', type=int, default=dim, help='output features size')
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

# seq2seq common
parser.add_argument('--dilation_rate', type=int, default=2, help='encoder input size')
parser.add_argument('--num_levels', type=int, default=3, help='decoder input size')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='teacher_forcing_ratio')
parser.add_argument('--importance', type=bool, default=False, help='importance')
# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=
                    32, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=1, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=24, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers

parser.add_argument('--embed_type', type=int, default=3, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=dim, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=dim, help='decoder input size')
parser.add_argument('--c_out', type=int, default=dim, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)

parser.add_argument('--dropout', type=float, default=0.05, help='dropout') #was 0.05
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu') #was true (MODIFY AutoCorrelation to add .cuda())
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]



if __name__ == '__main__':

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
# For plot in Matlab
"""
import numpy as np
data = np.load('C:\\Users\\Lyes\\PycharmProjects\\Coral\\LTSF\\ExoskeletonLoader\\Results\\60\\TempoNet\\pred.npy')
from scipy import io
io.savemat('C:\\Users\\Lyes\\PycharmProjects\\Coral\\LTSF\\ExoskeletonLoader\\Results\\60\\TempoNet\\pred.mat', {'data': data})
import numpy as np
data = np.load('C:\\Users\\Lyes\\PycharmProjects\\Coral\\LTSF\\ExoskeletonLoader\\Results\\60\\TempoNet\\true.npy')
from scipy import io
io.savemat('C:\\Users\\Lyes\\PycharmProjects\\Coral\\LTSF\\ExoskeletonLoader\\Results\\60\\TempoNet\\true.mat', {'data': data})

from scipy import io
io.savemat('C:\\Users\\Lyes\\PycharmProjects\\Coral\\LTSF\\ExoskeletonLoader\\Results\\60\\TempoNet\\true.mat', {'act': actl_[model]})
io.savemat('C:\\Users\\Lyes\\PycharmProjects\\Coral\\LTSF\\ExoskeletonLoader\\Results\\60\\TempoNet\\true.mat', {'act': pred_[model]})

io.savemat('C:\\Users\\Lyes\\PycharmProjects\\Coral\\LTSF\\ExoskeletonLoader\\Results\\60\\TempoNet\\true.mat', {'act': actl_[model]})
io.savemat('C:\\Users\\Lyes\\PycharmProjects\\Coral\\LTSF\\ExoskeletonLoader\\Results\\60\\TempoNet\\true.mat', {'act': pred_[model]})
"""