# 🦾 FocalGatedNet & Unified Transformer Family for Time-Series Forecasting

A unified **PyTorch pipeline** for **knee joint angle forecasting** and general **multivariate time-series prediction**, supporting **FocalGatedNet**, **TempoFormer**, **PatchTST**, **Autoformer**, **Informer**, and vanilla **Transformer** baselines.

> 🏆 Our paper *“Improving Knee Joint Angle Prediction through Dynamic Contextual Focus and Gated Linear Units (FocalGatedNet)”* is published in **Computers in Biology and Medicine (Elsevier)** — **CiteScore 2024: 13.0, ranked Top 6% globally (94th percentile in Health Informatics, 92nd percentile in Computer Science Applications)**.  
> DOI: [10.1016/j.compbiomed.2025.111119](https://doi.org/10.1016/j.compbiomed.2025.111119)

> ⚠️ **Note:** I am still uploading the full set of scripts, datasets, and resources.  
> 🔗 Pretrained **checkpoint models** can be found here: [Google Drive Link](YOUR_GOOGLE_DRIVE_LINK)

---

## 🔍 Overview

This repository provides a **single entry script** that standardizes:

- 📦 Data loading (CSV), experiment configs, training/testing loops  
- 🧩 Model zoo selection via `--model` flag  
- 🏃 Robust defaults for **exoskeleton-based gait analysis** (e.g., `--target knee_sagittal`)  
- 🔒 Reproducibility (global seeds) & easy GPU / multi-GPU switching  
- 🔬 Training, testing, and forecasting via a unified `Exp_Main` wrapper

---

## ✨ Supported Models

Choose with `--model`:

- **FocalGatedNet** (Dynamic Contextual Focus + GLU; accurate & noise-robust)
- **TempoFormer**, **TempoFormerModified**, **TempoNet**
- **Autoformer**, **Informer**, **Transformer**
- **EDLstm** (LSTM baseline)
- *(Legacy)* LSTNet / TCN via shared args

Add your own model by implementing it and registering in `Exp_Main`.

---

## 📁 Project Structure

repo/
├─ exp/
│ └─ exp_main.py # experiment wrapper (train/test/predict)
├─ checkpoints/ # saved weights
├─ dataset/
│ └─ selected_features_advanced.csv
├─ scripts/ # (optional) helper scripts
├─ main.py # entry script
└─ README.md

---

## ⚙️ Installation

```bash
# create env (example)
conda create -n gait python=3.10 -y
conda activate gait

# install PyTorch (choose CUDA/CPU as needed)
# e.g., CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# core deps
pip install numpy pandas scikit-learn matplotlib tqdm
```

If you use AMP (--use_amp), ensure your CUDA/driver stack supports it.

---

## 📊 Performance Highlights

| Forecast Horizon | MAE ↓ | RMSE ↓ | MAPE ↓ | R² ↑ |
|------------------|-------|--------|--------|------|
| **20 ms**        | **0.681** | **0.837** | **0.085** | **99.84%** |
| **40 ms**        | **0.985** | **1.070** | **0.120** | **99.65%** |
| **60 ms**        | **1.058** | **1.404** | **0.129** | **99.54%** |
| **80 ms**        | **1.115** | **1.584** | **0.119** | **99.41%** |
| **100 ms**       | **1.234** | **1.794** | **0.140** | **99.25%** |

FocalGatedNet consistently outperforms LSTM, GRU, Transformer, Autoformer, Informer, DLinear, and NLinear across short and long-term horizons.

---

## 🚀 Quick Start

### Test Only
```bash
python main.py   --model FocalGatedNet   --model_id FGNet_Test   --data custom   --root_path ./dataset   --data_path selected_features_advanced.csv   --features M --target knee_sagittal   --seq_len 128 --label_len 64 --pred_len 100   --checkpoints ./checkpoints
```

### Train → Test
```bash
python main.py   --is_training 1   --model FocalGatedNet   --model_id FGNet_Train01   --data custom   --root_path ./dataset   --data_path selected_features_advanced.csv   --features M --target knee_sagittal   --seq_len 128 --label_len 64 --pred_len 100   --train_epochs 20 --batch_size 32 --learning_rate 1e-4   --patience 5 --des exp01   --checkpoints ./checkpoints
```

### Predict Future (optional)

Add `--do_predict` to the training run to save future forecasts.

---

## 🧪 Ablation Study

| Model Variant      | MAE (80 ms) ↓ | RMSE (80 ms) ↓ |
|--------------------|---------------|----------------|
| Base (no GLU/DCF)  | 12.36         | 13.82          |
| + GLU only         | 1.51          | 1.92           |
| **+ GLU + DCF**    | **1.12**      | **1.58**       |

GLU improves short-term accuracy, while GLU + DCF ensures superior long-term performance.

---

## 🔧 Key Arguments (Most Used)

| Argument | Default | Meaning |
|----------|---------|---------|
| `--model` | TempoFormer | Model name (EDLstm, TempoFormer, TempoFormerModified, TempoNet, Autoformer, Informer, Transformer, FocalGatedNet, PatchTST) |
| `--model_id` | NewModelx2 | Tag for logs/checkpoints |
| `--is_training` | 0 | 1=train+test (+predict if set), 0=test only |
| `--data` | custom | Dataset type label |
| `--root_path` | ./dataset/ | Folder for data |
| `--data_path` | selected_features_advanced.csv | CSV file |
| `--features` | M | M=multi→multi, S=uni→uni, MS=multi→uni |
| `--target` | knee_sagittal | Target column |
| `--seq_len` | 128 | Input window length |
| `--label_len` | 64 | “Start token” length |
| `--pred_len` | 100 | Forecast horizon |
| `--train_epochs` | 20 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--patience` | 5 | Early stopping |
| `--use_amp` | False | Mixed precision |
| `--use_gpu` | True | Use CUDA if available |
| `--use_multi_gpu` | False | Enable multi-GPU |
| `--devices` | "0,1,2,3" | Device IDs when multi-GPU |

Transformer knobs: `--d_model 512 --n_heads 8 --e_layers 4 --d_layers 3 --d_ff 2048 --factor 3 --dropout 0.05 --embed timeF --distil True`  
PatchTST knobs: `--patch_len 32 --stride 8 --revin 1 --decomposition 1 --kernel_size 24`

---

## 🧬 Reproducibility

```python
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
```

For strict determinism:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

And optionally in Python:

```python
torch.use_deterministic_algorithms(True)
```

---

## ⚡ Tips

🖥️ **GPU:** Ensure drivers/CUDA match the PyTorch build.  
🚀 **Multi-GPU:** Use `--use_multi_gpu --devices "0,1,2,3"` (first ID = master GPU).  
📈 **Long horizons:** Increase `--seq_len`, tune `--d_model`, consider `--revin 1` for PatchTST.  
🌪️ **Noise robustness:** FocalGatedNet is designed for sensor noise; tune dropout & normalization.

---

## 📝 Citation

```bibtex
@article{SaadSaoud2025FocalGatedNet,
  title   = {Improving Knee Joint Angle Prediction through Dynamic Contextual Focus and Gated Linear Units (FocalGatedNet)},
  author  = {Lyes Saad Saoud and Humaid Ibrahim and Ahmad Aljarah and Irfan Hussain},
  journal = {Computers in Biology and Medicine},
  year    = {2025},
  doi     = {10.1016/j.compbiomed.2025.111119}
}
```

---

## 🤝 Acknowledgments

Grateful to **Khalifa University**, **KU Center for Autonomous Robotic Systems (KUCARS)**, and the **Advanced Research and Innovation Center (ARIC)** for their invaluable support and collaboration.
