## 1. VERDICT

**NOT SAFE FOR LIVE TRADING**

This system is structurally compromised by fatal inconsistencies between its training, testing, and execution environments. The most egregious flaw is the **Unclosed Bar Execution** in the live bridge, where the agent trades on incomplete candle data. Furthermore, a **Model Architecture Mismatch** ensures the testing script will instantly crash, as it attempts to load an MLP-based `MaskablePPO` model into a `RecurrentPPO` container. Finally, the reliance on Yahoo Finance for H1 Forex data guarantees that the backtested spreads and stop-loss mechanics are completely detached from real-world broker dynamics. Real capital would be rapidly depleted by invisible transaction costs and execution slippage.

## 2. CRITICAL FAILURES

* **Issue: Live Unclosed Bar Execution (Future Leakage)**
    * **Failure Mode:** `live_bridge.py` fetches data using `mt5.copy_rates_from_pos(..., 0, n_bars)`. Index `0` in MT5 is the *currently forming*, incomplete bar.
    * **Real-world consequence:** The bot wakes up, fetches a bar that has been open for 5 seconds, treats it as a fully closed H1 candle, recalculates moving averages and RSI based on this noise, and executes a trade. It is hallucinating market closes.
    * **Severity:** FATAL.
    * **Fix:** Change MT5 fetch logic to `mt5.copy_rates_from_pos(..., 1, n_bars)` to guarantee only mathematically closed bars are processed.

* **Issue: Model Architecture / Inference Crash**
    * **Failure Mode:** `train_agent.py` trains a `MaskablePPO` model using `MlpPolicy`. However, `test_agent.py` attempts to load this exact model file using `RecurrentPPO.load(...)` and passes `lstm_states`. 
    * **Real-world consequence:** The test evaluation script will immediately crash upon trying to load the model due to missing recurrent weight dictionaries. The test metrics provided are likely fabricated or from a previous, overwritten iteration.
    * **Severity:** FATAL.
    * **Fix:** Standardize the architecture. If action masking is paramount, strictly use `MaskablePPO` across all scripts. If memory is required, use `RecurrentPPO` and manually implement action logit-masking.

* **Issue: Reward Signal Annihilation (Scale Mismatch)**
    * **Failure Mode:** The reward function is $R_t = \ln(TotalEquity_t / TotalEquity_{t-1})$. With a 0.01 micro-lot on a \$1,000 account, a 10-pip win alters equity to \$1,001. $\ln(1.001) \approx 0.00099$. 
    * **Real-world consequence:** The reward signal (~0.001) is completely overwhelmed by PPO's entropy coefficient (`ent_coef = 0.01`). The optimizer will prioritize random exploration over exploiting profitable trades, resulting in a model that learns nothing but noise.
    * **Severity:** FATAL.
    * **Fix:** Scale the reward signal explicitly, e.g., $R_t = (UnrealizedPnL_{pips} / ATR) * 10$, or multiply the log return by a factor of 10,000.

## 3. HIGH / MEDIUM ISSUES

* **Issue: Garbage Data Source for Forex Training (High)**
    * **Failure Mode:** `download_multi_data.py` uses `yfinance` for Forex. Yahoo Finance provides indicative mid-pricing, lacks true bid/ask spreads, and its "Volume" metric is fictional.
    * **Real-world consequence:** A 10-pip stop-loss in the backtest will survive, but in live MT5 execution, the actual bid/ask spread (especially during 21:00 UTC rollover) will instantly trigger the stop-loss.
    * **Fix:** Replace `yfinance` with Dukascopy or TrueFX tick data. Downsample to H1 with explicit Bid-OHLC and Ask-OHLC.

* **Issue: Static Position Sizing vs. Dynamic Risk (High)**
    * **Failure Mode:** `lot_size = 0.01` is hardcoded regardless of the stop-loss distance. An 80-pip SL risks 8x more capital than a 10-pip SL.
    * **Real-world consequence:** The RL agent will likely learn to exclusively use the 80-pip SL because it avoids being stopped out, creating a massive negative skew in the risk profile. A few losses will wipe out weeks of gains.
    * **Fix:** Implement Volatility-Adjusted Sizing: `lots = (Equity * 0.01) / (SL_pips * pip_value)`.

* **Issue: Global Scaler Leakage & Non-Stationarity (Medium)**
    * **Failure Mode:** `FeatureEngine` uses `StandardScaler`. Market volatility expands and contracts (heteroskedasticity). A global mean/variance scaler over years of data causes recent high-volatility events to look like extreme outliers.
    * **Real-world consequence:** The model's neural network will see activation saturation (z-scores > 5) during volatile regimes, paralyzing its decision-making exactly when risk management is most critical.
    * **Fix:** Use a rolling Z-score normalization (e.g., rolling 200-period mean/std) or strictly stick to scale-invariant indicators like RSI and normalized ATR.

## 4. BACKTEST VALIDITY

**INVALID AND MISLEADING**

The backtest environment (`trading_env.py`) executes trades at the exact `Close` of the bar that generated the signal, assuming zero latency. Furthermore, it assumes a static spread (`spread_pips=1.0`) and uniform slippage. In live forex markets, spreads widen dramatically during news events and the daily rollover. Because the backtest data (`yfinance`) lacks this microstructure, the metrics outputted by `evaluate_oos.py` are mathematically impossible to replicate in production.

## 5. MODEL / RL ANALYSIS

* **State Space:** The inclusion of `unrealised_pnl_norm` introduces an extreme "disposition effect" bias. RL models given unrealized PnL often learn to hold losing trades indefinitely to avoid taking a realized negative reward, hoping it returns to breakeven.
* **Action Space:** Discretizing SL and TP into static arrays (`[10, 20, 40, 80]`) limits the model. 10 pips on GBPJPY is noise; 10 pips on EURCHF is a major move. Actions should be multipliers of the current ATR (e.g., $SL = 1.5 \times ATR$).
* **Stability:** `train_agent.py` applies `ActionMasker` properly, but PPO's value function will struggle to predict returns accurately because the duration of a trade (and thus the discount factor $\gamma$) is highly variable.

## 6. RISK MANAGEMENT GAP

* **Missing Volatility Sizing:** Risking \$1 on a trade with a 10-pip SL and \$8 on an 80-pip SL destroys risk parity.
* **Missing Exposure Limits:** The bot can open a position on `EURUSD` and `GBPUSD` simultaneously, doubling USD exposure without correlation awareness.
* **Missing News Filter:** Technical indicators are useless during Non-Farm Payrolls (NFP) or FOMC. There is no API integration to pause trading during Tier-1 economic releases.
* **Missing MT5 Order Type Validation:** `live_bridge.py` uses `mt5.ORDER_FILLING_IOC`. Many ECN brokers strictly require `mt5.ORDER_FILLING_FOK` (Fill or Kill) or `RETURN`. *UNCERTAIN — NEEDS VERIFICATION based on the specific broker.*

## 7. GPU / CPU ANALYSIS

* **Is GPU actually used?** Yes, during training in `train_agent.py`.
* **Is fallback correct?** **VALID.** `device_utils.py` correctly tests actual tensor execution (`torch.tanh(x).cuda()`). This is a highly robust check that catches "No kernel image" errors on newer architectures like the RTX 50xx series.
* **Is architecture optimal?** Yes. `live_bridge.py` explicitly forces `device="cpu"` when loading the model. GPU inference for a batch size of 1 on an H1 timeframe introduces unnecessary PCI-E transfer latency and instability. CPU is the correct choice here.

## 8. TOP 1% GAP

Elite institutional trading systems do not rely on Gym environments running discrete pip actions. The gaps include:
1.  **Microstructure Simulation:** Top-tier systems use historical order book data (Level 2) or tick data to simulate queue position and realistic adverse selection, not `np.random.uniform` slippage.
2.  **Continuous Action Spaces with Beta-Sizing:** Using continuous distributions (e.g., SAC or TD3) to output exact position sizes and exit boundaries based on Kelly Criterion limits.
3.  **Walk-Forward Optimization:** Validating across rolling out-of-sample windows to detect strategy decay, rather than a single 15% chronological split.
4.  **Shadow Mode Engine:** A parallel live bridge that records exactly what the bot *would* have executed, comparing shadow fills vs. backtest fills to measure execution drift.

## 9. REBUILD BLUEPRINT

* **Data Layer:** Ditch `yfinance`. Integrate `TickData` from Dukascopy. Resample to 1-minute (M1) bars for the backtest step loop, but feed H1 features to the model. This allows exact intra-bar SL/TP triggering.
* **Feature Layer:** Remove global `StandardScaler`. Implement rolling window Z-scores: $Z_t = (X_t - \mu_{t-200}) / \sigma_{t-200}$. Replace static pip actions with ATR multiples.
* **Model Layer:** Standardize on `MaskablePPO`. Rescale the reward function by multiplying base returns by $10^4$ (basis points) so gradients don't vanish. Remove `unrealised_pnl` from the observation space to prevent holding bias.
* **Execution Layer:** Rewrite `live_bridge.py` to trigger on a precise cron-job scheduler synchronized to an NTP server, requesting bar `1` (closed bar) from MT5.
* **Risk Engine:** Implement dynamic fractional Kelly sizing: Risk strictly 1% of current `equity_usd` per trade, adjusting `lot_size` dynamically based on the SL distance.

## 10. PRIORITY ACTION PLAN

1.  **FIX MT5 DATA LEAK:** Update `mt5.copy_rates_from_pos` in `live_bridge.py` to fetch from index `1` instead of `0`.
2.  **RESOLVE MODEL CRASH:** Rewrite `test_agent.py` to load `MaskablePPO` instead of `RecurrentPPO`.
3.  **RESCALE REWARDS:** Multiply the log-return reward in `trading_env.py` by 10,000 so the agent can overcome the entropy coefficient.
4.  **REPLACE DATA SOURCING:** Scrap `download_multi_data.py`. Download institutional-grade tick data with Bid/Ask spreads.
5.  **IMPLEMENT DYNAMIC LOT SIZING:** Calculate `lot_size` dynamically in `live_bridge.py` based on SL pips to ensure uniform USD risk per trade.
6.  **CONVERT TO ROLLING SCALER:** Modify `FeatureEngine` to use rolling Z-scores instead of `sklearn.StandardScaler` to handle market heteroskedasticity.
7.  **UPGRADE BACKTEST RESOLUTION:** Modify `trading_env.py` to step through M1 data while providing H1 observations to accurately model intra-bar stop-loss hits.
8.  **ADD ECONOMIC CALENDAR API:** Implement a hard pause in `live_bridge.py` 30 minutes before and after high-impact news events.
9.  **VERIFY BROKER FILL TYPES:** Test `mt5.ORDER_FILLING_IOC` on the live demo account; fall back to `FOK` if rejected by the broker.
10. **IMPLEMENT SHADOW TRADING:** Run the live bridge in a read-only logging mode for 2 weeks to calculate the true divergence between simulated MT5 slippage and actual market microstructure.