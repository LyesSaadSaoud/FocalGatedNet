# FocalGatedNet: A Novel Deep Learning Model for Accurate Knee Joint Angle Prediction
Humaid Ibrahim, Lyes Saad Saoud, Ahmad Aljarah, Irfan Hussain

This repo contains the supported files to reproduce the [paper]([https://arxiv.org/abs/2306.06900]) results using the stationary wavelet transform and deep transfomers 

## ABSTRACT
Predicting knee joint angles accurately is critical for biomechanical analysis and rehabilitation. This paper introduces a new deep learning model called FocalGatedNet that incorporates Dynamic Contextual Focus (DCF) Attention and Gated Linear Units (GLU) to enhance feature dependencies and interactions. Our proposed model is evaluated on a large-scale dataset and compared to existing models such as Transformer, Autoformer, Informer, NLinear, DLinear, and LSTM in multi-step gait trajectory prediction. Our results demonstrate that FocalGatedNet outperforms other state-of-the-art models for long-term prediction lengths (60 ms, 80 ms, and 100 ms), achieving an average improvement of 13.66% in MAE and 8.13% in RMSE compared to the second-best performing model (Transformer). Furthermore, our model has a lower computational load than most equivalent deep learning models. These results highlight the effectiveness of our proposed model for knee joint angle prediction and the importance of our modifications for this specific application.




The proposed deep transformer SWT model for the household power consumption forecasting.

## Getting started
1. Install Python 3.9, PyTorch 1.9
2. Download all files and put them in the same folder. 
3. cd to path. 
4. Run the file Coming soon

@misc{ibrahim2023focalgatednet,
      title={FocalGatedNet: A Novel Deep Learning Model for Accurate Knee Joint Angle Prediction}, 
      author={Humaid Ibrahim and Lyes Saad Saoud and Ahmad Aljarah and Irfan Hussain},
      year={2023},
      eprint={2306.06900},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}

Emails: lyes.saoud@ku.ac.ae


