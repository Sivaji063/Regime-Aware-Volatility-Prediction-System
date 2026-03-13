# 📈 Regime-Aware Volatility Prediction System

> **Final Year Research Project — Department of Quantitative Finance & Deep Learning**
> Using Self-Supervised LSTM to detect volatility regimes and drive risk-adjusted trading decisions.

---

## 🔍 Overview

This project builds an end-to-end regime-aware trading system that:

- Uses **Self-Supervised LSTM Pretraining** via masked time-series reconstruction
- Classifies market into **High / Low Volatility Regimes** without labelled data
- Applies **Dynamic Position Sizing** based on predicted regime
- Achieves **82% Accuracy** and **0.892 ROC-AUC** on held-out test data
- Validated across **5 Walk-Forward Folds** including COVID crash & 2022 bear market
- Supports **Indian Markets (NSE & BSE)** with live IST timezone predictions

---

## 📊 Results

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Calmar Ratio |
|----------|-------------|-------------|-------------|-------------|
| **Regime-Aware** | **34.21%** | **0.923** | **-8.43%** | **1.171** |
| Momentum Only | 28.43% | 0.812 | -11.23% | 0.742 |
| Buy & Hold | 51.23% | 0.934 | -23.45% | 0.563 |

### Walk-Forward Validation

| Fold | Period | Strat Sharpe | Mom Sharpe | Accuracy |
|------|--------|-------------|-----------|---------|
| 1 | 2019-01 → 2019-10 | 0.823 | 0.712 | 79.1% |
| 2 | 2019-11 → 2020-08 | 0.912 | 0.634 | 81.2% |
| 3 | 2020-09 → 2021-06 | 0.743 | 0.698 | 77.8% |
| 4 | 2021-07 → 2022-04 | 1.021 | 0.821 | 83.4% |
| 5 | 2022-05 → 2023-02 | 0.934 | 0.756 | 82.1% |
| **Mean** | | **0.887** | **0.724** | **80.7%** |

---

## 🏗️ System Architecture

```
Raw OHLCV Data
      │
      ▼
Feature Engineering (9 features)
      │
      ▼
Self-Supervised Pretraining
(Masked LSTM Reconstruction — no labels needed)
      │
      ▼
Fine-Tuning
(Binary Regime Classifier — High / Low Vol)
      │
      ▼
Trading Strategy
(Momentum Signal + Regime Position Sizing)
      │
      ▼
Backtesting & Walk-Forward Validation
```

---

## 📐 Model Architecture

```
Input: (batch, 30 days, 9 features)
       │
       ▼
LSTMEncoder
  ├── LSTM Layer 1 (hidden=128)
  ├── Dropout (0.3)
  └── LSTM Layer 2 (hidden=128)
       │
       ▼  (Pretraining)          ▼  (Fine-tuning)
   Decoder Head              Classifier Head
   (Reconstruct              (Binary: High/Low)
    masked values)
```

---

## 🗂️ Features Used

| Feature | Definition | Rationale |
|---------|-----------|-----------|
| Daily Return | Pt / Pt-1 - 1 | Primary price signal |
| Log Return | ln(Pt / Pt-1) | Normalised returns |
| Vol-20 | Std(returns, 20d) | Core regime signal |
| Vol-5 | Std(returns, 5d) | Short-term volatility |
| MA Ratio | MA10 / MA20 | Trend crossover |
| RSI-14 | Wilder RSI | Overbought/oversold |
| Volume Change | Vt / Vt-1 - 1 | Liquidity signal |
| HL Range | (High-Low) / Close | Intraday volatility |
| Momentum-10 | Pt / Pt-10 - 1 | 10-day momentum |

---

## 💹 Trading Strategy

```
Signal    : Long when 10-day momentum > 0, else Cash
Sizing    : 100% capital in Low Vol Regime
            50%  capital in High Vol Regime
Cost      : 0.1% transaction cost per trade
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10 | Core language |
| PyTorch | 2.1 | Deep learning |
| yfinance | 0.2+ | Market data |
| scikit-learn | 1.1+ | Preprocessing & metrics |
| pandas | 1.5+ | Data manipulation |
| numpy | 1.23+ | Numerical computing |
| matplotlib | 3.6+ | Visualisation |
| seaborn | 0.12+ | Statistical plots |
| pytz | 2022.7+ | IST timezone support |

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

```
1. Go to https://colab.research.google.com
2. File → Upload notebook OR open regime_volatility_colab_cells.py
3. Runtime → Change runtime type → GPU
4. Run cells Cell 1 to Cell 24 in order
5. Change TICKER in Cell 3 to your preferred stock
```

### Option 2 — Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/regime-aware-volatility-prediction.git
cd regime-aware-volatility-prediction

# Install dependencies
pip install -r requirements.txt

# Run the code
python regime_volatility_colab_cells.py
```

---

## 📈 Supported Tickers

### Indian Markets (NSE & BSE)

| Stock | NSE | BSE |
|-------|-----|-----|
| NIFTY 50 | `^NSEI` | — |
| SENSEX | — | `^BSESN` |
| Reliance | `RELIANCE.NS` | `RELIANCE.BO` |
| TCS | `TCS.NS` | `TCS.BO` |
| Infosys | `INFY.NS` | `INFY.BO` |
| HDFC Bank | `HDFCBANK.NS` | `HDFCBANK.BO` |
| Wipro | `WIPRO.NS` | `WIPRO.BO` |
| SBI | `SBIN.NS` | `SBIN.BO` |

### US Markets

| Index/ETF | Ticker |
|-----------|--------|
| S&P 500 ETF | `SPY` |
| NASDAQ | `QQQ` |
| Dow Jones | `DIA` |

---

## 📁 Project Structure

```
regime-aware-volatility-prediction/
│
├── 📄 README.md                              ← This file
├── 📄 requirements.txt                       ← Python dependencies
├── 📄 .gitignore                             ← Files to ignore
├── 🐍 regime_volatility_colab_cells.py       ← Main code (24 cells)
└── 📝 Regime_Aware_Volatility_Research_Paper.docx ← Research paper
```

---

## 🔢 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Sequence Length | 30 days |
| Masking Ratio | 20% |
| LSTM Hidden Size | 128 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Pretrain Epochs | 40 |
| Finetune Epochs | 30 |
| Pretrain LR | 1e-3 (cosine) |
| Finetune LR | 5e-4 → 3e-4 |
| Batch Size | 64 |
| Optimiser | AdamW |
| Transaction Cost | 0.1% |

---

## 📚 References

- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica*.
- Devlin et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL-HLT*.
- Fischer & Krauss (2018). Deep learning with LSTM for financial predictions. *EJOR*.
- Sharpe, W. F. (1994). The Sharpe ratio. *Journal of Portfolio Management*.
- Zerveas et al. (2021). Transformer-based framework for time series learning. *KDD*.

---

## 👤 Author

**Final Year Student**
Department of Quantitative Finance & Deep Learning
March 2026

---

## ⭐ If this project helped you, please give it a star!
