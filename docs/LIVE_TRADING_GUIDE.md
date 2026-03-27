# Live Trading Strategy Guide: MT5 / OANDA
# ==========================================
# This guide explains how to take the trained MaskablePPO model
# into live production on MetaTrader 5 (MT5) or OANDA.

## 1. Architecture Overview

```
[MT5 / OANDA API]
       |
       v
[live_bridge.py]  ← fetches latest H1 bar every hour
       |
       v
[FeatureEngine.push(bar)]  ← computes 15 indicators (same as training)
       |
       v
[MaskablePPO.predict(obs, action_masks)]  ← GPU inference
       |
       v
[mt5.order_send() / OANDA REST API]  ← executes trade
       |
       v
[live_state.json]  ← saves position/equity for crash recovery
```

## 2. Prerequisites

### Packages (add to requirements.txt)
```bash
pip install MetaTrader5       # Windows only, MT5 terminal must be installed
pip install oandapyV20        # For OANDA REST API alternative
```

### Account Setup
- **MT5**: Open a demo account at your broker (e.g. ICMarkets, Pepperstone)
  - Enable Algo Trading in MT5 → Tools → Options → Expert Advisors
  - Use `mt5.initialize(login=YOUR_LOGIN, password="...", server="ICMarkets-Demo")` in `live_bridge.py`
- **OANDA**: Get API key from https://www.oanda.com/demo-account/

## 3. Running the Live Bridge (MT5)

```bash
# Step 1: Make sure model is trained
python train_agent.py

# Step 2: Start live bridge (runs indefinitely, 1 decision per H1 bar)
python live_bridge.py
```

### What happens on each bar:
1. Fetches 200 H1 bars from MT5 to warm up the FeatureEngine
2. Generates a 15-feature observation using the SAME scaler fitted during training
3. Generates the action mask (OPEN blocked if in trade, CLOSE blocked if flat)
4. Model predicts: HOLD / CLOSE / OPEN(direction, SL, TP)
5. Sends order via `place_order()` (implement `mt5.order_send()` in that function)
6. Saves state to `live_state.json` — if the script crashes or restarts, it resumes exactly where it left off

## 4. Implementing `place_order()` for MT5

Replace the placeholder `print` in `live_bridge.py`:

```python
import MetaTrader5 as mt5

def place_order(symbol, action_type, sl_pips=20, tp_pips=40, lot=0.01):
    pip = 0.0001 if "JPY" not in symbol else 0.01
    price = mt5.symbol_info_tick(symbol).ask if action_type == "BUY" else mt5.symbol_info_tick(symbol).bid
    
    if action_type == "BUY":
        sl = price - sl_pips * pip
        tp = price + tp_pips * pip
        order_type = mt5.ORDER_TYPE_BUY
    elif action_type == "SELL":
        sl = price + sl_pips * pip
        tp = price - tp_pips * pip
        order_type = mt5.ORDER_TYPE_SELL
    elif action_type == "CLOSE":
        # Close all open positions for this symbol
        positions = mt5.positions_get(symbol=symbol)
        for pos in positions:
            close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            close_price = mt5.symbol_info_tick(symbol).bid if pos.type == 0 else mt5.symbol_info_tick(symbol).ask
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                "volume": pos.volume, "type": close_type,
                "position": pos.ticket, "price": close_price,
                "deviation": 20, "magic": 234000,
                "comment": "RL Bot Close", "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            mt5.order_send(close_request)
        return

    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
        "volume": lot, "type": order_type,
        "price": price, "sl": sl, "tp": tp,
        "deviation": 20, "magic": 234000,
        "comment": "RL Bot", "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Order failed: {result.retcode} - {result.comment}")
    else:
        print(f"[ORDER] {action_type} {lot} lot at {price:.5f}, SL={sl:.5f}, TP={tp:.5f}")
```

## 5. Risk Management Rules

| Rule | Recommended Value | Reason |
|------|------------------|--------|
| Lot size | 0.01 (micro) | Start small on demo first |
| Max DD before stop | 10% | Stop bot if equity drops 10% from peak |
| Daily loss limit | 3% | Broker protection |
| Max open trades | 1 | The model is trained single-position |
| JPY pip multiplier | 0.01 | Not 0.0001 |

## 6. OANDA Alternative

```python
import oandapyV20
from oandapyV20.endpoints import orders, positions

client = oandapyV20.API(access_token="YOUR_TOKEN", environment="practice")

def oanda_buy(instrument="EUR_USD", units=1000, sl_pips=20, tp_pips=40):
    pip = 0.0001
    current_price = float(get_oanda_price(instrument))
    data = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),
            "stopLossOnFill": {"price": f"{current_price - sl_pips*pip:.5f}"},
            "takeProfitOnFill": {"price": f"{current_price + tp_pips*pip:.5f}"},
        }
    }
    r = orders.OrderCreate(accountID="YOUR_ACCOUNT_ID", data=data)
    client.request(r)
```

## 7. Production Checklist

- [ ] Run on demo for at least 30 days before going live
- [ ] Verify `scaler_features.pkl` matches the model version
- [ ] Set up a scheduler (Windows Task Scheduler / cron) to restart `live_bridge.py` on crash
- [ ] Monitor `live_state.json` — if `equity_usd` drops below threshold, halt the bot
- [ ] Log every decision to a CSV for post-analysis
- [ ] Never risk more than 1-2% per trade on a live account
