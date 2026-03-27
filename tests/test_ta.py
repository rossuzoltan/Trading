import pandas as pd
import pandas_ta as ta

df = pd.DataFrame({
    "Open": [1, 2, 3, 4, 5] * 20,
    "High": [2, 3, 4, 5, 6] * 20,
    "Low": [0, 1, 2, 3, 4] * 20,
    "Close": [1.5, 2.5, 3.5, 4.5, 5.5] * 20,
    "Volume": [100, 200, 300, 400, 500] * 20
})

df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.bbands(length=20, std=2, append=True)
df.ta.adx(length=14, append=True)

print(df.columns)
