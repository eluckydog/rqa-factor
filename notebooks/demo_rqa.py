"""Demo: RQA factor on synthetic market data."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rqa_factor.factor import compute_rqa_factors

np.random.seed(42)

# Synthetic data: 3 regimes
T = 504  # ~2 years
N = 10   # 10 stocks

returns = np.random.randn(T, N) * 0.015  # normal regime

# Crisis: higher volatility, more synchronized (months 200-300)
returns[200:300] += np.random.randn(100, N) * 0.03
common_shock = np.random.randn(100, 1) * 0.04
returns[200:300] += common_shock * np.ones((1, N)) * 0.8

# Recovery: lower volatility (months 350-450)
returns[350:450] *= 0.5

factors = compute_rqa_factors(returns, window=63, dim=3, tau=1, pct=10.0)

print("=== RQA Factor Demo ===")
print(f"Returns shape: {returns.shape}")
print(f"Factors length: {len(factors['rr'])} (windows of 63 days)")
print()

print("Factor stats (over 2yr synthetic data):")
for k in ['rr', 'det', 'lmax', 'entr']:
    vals = factors[k][~np.isnan(factors[k])]
    print(f"  {k:6s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
          f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

# Signal comparison: before, during, after crisis
pre = slice(0, 137)   # before crisis (window=63 aligns end)
crisis = slice(137, 237)  # during crisis
post = slice(237, None)  # after

print()
print("DET (Determinism) by regime:")
print(f"  Pre-crisis:  {np.nanmean(factors['det'][pre]):.4f}")
print(f"  During:      {np.nanmean(factors['det'][crisis]):.4f}")
print(f"  Post-crisis: {np.nanmean(factors['det'][post]):.4f}")

print()
print("Lmax (longest diagonal line) by regime:")
print(f"  Pre-crisis:  {np.nanmean(factors['lmax'][pre]):.1f}")
print(f"  During:      {np.nanmean(factors['lmax'][crisis]):.1f}")
print(f"  Post-crisis: {np.nanmean(factors['lmax'][post]):.1f}")
