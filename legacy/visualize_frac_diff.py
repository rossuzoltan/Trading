import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engine import _frac_diff_weights, _apply_frac_diff

def main():
    # 1. Generate synthetic trending series with noise
    np.random.seed(42)
    n = 1000
    prices = 1.1 + np.cumsum(np.random.normal(0.0001, 0.001, n))
    
    # 2. Apply Fractional Differentiation (d=0.3)
    d = 0.3
    fd = _apply_frac_diff(pd.Series(prices), d)
    
    # 3. Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(prices, label=f"Raw Price (Non-Stationary)", color="blue")
    ax1.set_title("Institutional Alpha Diagnosis: Memory vs. Stationarity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(fd, label=f"FracDiff (d={d}, Stationary)", color="green")
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_title(f"Fractional Differentiation (d={d}) - Preserves Long Memory")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("frac_diff_diagnosis.png")
    print("Diagnosis plot saved -> frac_diff_diagnosis.png")

if __name__ == "__main__":
    main()
