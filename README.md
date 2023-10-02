# Improving Knee Joint Angle Prediction through Dynamic Contextual Focus and Gated Linear Units
Lyes Saad Saoud, Humaid Ibrahim, Ahmad Aljarah, Irfan Hussain

This repo contains the supported files to reproduce the [paper]([https://arxiv.org/abs/2306.06900]) results using the stationary wavelet transform and deep transfomers 

## ABSTRACT
Accurate knee joint angle prediction is crucial for biomechanical analysis and rehabilitation. In this study, we introduce FocalGatedNet, a novel deep learning model that incorporates Dynamic Contextual Focus (DCF) Attention and Gated Linear Units (GLU) to enhance feature dependencies and interactions. Our model is evaluated on a large-scale dataset and compared to established models in multi-step gait trajectory prediction.
Our results reveal that FocalGatedNet outperforms existing models for long-term prediction lengths (60 ms, 80 ms, and 100 ms), demonstrating significant improvements in Mean Absolute Error (MAE), Root Mean Square Error (RMSE) and Mean Absolute Percentage Error (MAPE). Specifically for the case of 80 ms, FocalGatedNet achieves a notable MAE reduction of up to 24\%, RMSE reduction of up to 14\%, and MAPE reduction of up to 36\% when compared to Transformer, highlighting its effectiveness in capturing complex knee joint angle patterns.
Moreover, FocalGatedNet maintains a lower computational load than most equivalent deep learning models, making it an efficient choice for real-time biomechanical analysis and rehabilitation applications.

## Getting started
1. Install Python 3.9, PyTorch 1.9
2. Download all files and put them in the same folder. 
3. cd to path. 
4. Run the file Coming soon

@misc{ibrahim2023focalgatednet,
      title={Improving Knee Joint Angle Prediction through Dynamic Contextual Focus and Gated Linear Units}, 
      author={Lyes Saad Saoud, Humaid Ibrahim, Ahmad Aljarah, Irfan Hussain},
      year={2023},
      eprint={2306.06900},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}

Emails: lyes.saoud@ku.ac.ae


