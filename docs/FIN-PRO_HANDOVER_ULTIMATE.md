# Ultimate Senior Quant Forex RL Handover (Phase 5)

## 🎯 Project Objective
Establish a high-frequency, professional-grade DRL trading agent for major Forex pairs using Reinforcement Learning (RecurrentPPO/LSTM) with a focus on risk-adjusted returns and hardware-accelerated training.

## 🏗 High-Level Architecture
- **Environment**: Custom `ForexTradingEnv` using **Delta-Equity Rewards** (Pure financial objective).
- **Features**: Stationary **Log Returns** and **Rolling Z-scores** (@ indicators.py).
- **Model**: **RecurrentPPO (LSTM)** in `sb3-contrib`.
- **Policy**: `MlpLstmPolicy` with `window_size = 1` (LSTM handles the memory).
- **Scale**: 16 parallel environments (`SubprocVecEnv`) across 4 symbols.

## 💻 Hardware Utilization (Optimized)
- **CPU**: 16 parallel processes for environment stepping (10x+ FPS boost).
- **GPU**: **RTX 5050 (8GB)** utilization via `device="cuda"`.
- **Network**: Deeper 512-neuron layers with a 2-layer LSTM backbone.

## 📊 Data Infrastructure
- **Dataset**: `data/FOREX_MULTI_SET.csv` (69k+ bars across EURUSD, GBPUSD, USDJPY, AUDUSD).
- **Split**: 70/15/15 Chronological Split (Train / Validation / Test).
- **Validation**: SB3 `EvalCallback` monitors the validation set every 10,000 steps to save `best_model.zip`.

## 🛠 Project Files
- `trading_env.py`: The core Gymnasium environment (Delta-Equity, Fixed WIN=1).
- `indicators.py`: Feature engineering (Log returns, Z-Normalization).
- `train_agent.py`: High-performance 16-way multi-asset training script.
- `test_agent.py`: Out-of-sample benchmarking.
- `live_bridge_example.py`: MT5 integration template.
- `download_multi_data.py`: Multi-symbol Yahoo Finance fetcher.

## 🚀 Next Steps (For the New Model)
1. **Verify CUDA**: Run `python -c "import torch; print(torch.cuda.is_available())"`. It should be `True` after the current installer finishes.
2. **Execute Training**: Run `python train_agent.py`. Expect **1000+ FPS** with the dual-GPU/Multi-core setup.
3. **Monitor Val**: Watch the `checkpoints/best_model.zip`. Use `tensorboard --logdir ./tensorboard_log/`.
4. **Live Bridge**: Once satisfied with the `equity_curve_phase3.png`, use the `live_bridge_example.py` to connect to a MetaTrader 5 Demo account.

## ⚠️ Critical Advice
- **LSTM Persistence**: DO NOT reset `lstm_states` mid-trade in live mode.
- **Concept Drift**: Retrain every 3-6 months as market regimes shift.
- **Reward Leakage**: Never go back to "Reward Shaping" (e.g., hold-rewards). Delta-Equity is the only way to ensure real profit.
