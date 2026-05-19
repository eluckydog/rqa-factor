# RQA Factor

> Recurrence Quantification Analysis for market regime detection.

Phase-space embedding of return time series → recurrence plots → nonlinear dynamical measures as regime-change factors. Based on Zbilut & Webber (1992) / Marwan et al. (2007).

## Observables

| Measure | Meaning | Crisis Signal |
|---------|---------|---------------|
| **RR** (Recurrence Rate) | Fraction of recurrent phase-space points | Often rises (patterns repeat) |
| **DET** (Determinism) | Fraction of recurrent points forming diagonal lines | Rises (system becomes more regular) |
| **Lmax** (Max diagonal) | Longest deterministic structure | Lengthens (periodicity) |
| **ENTR** (Shannon Entropy) | Entropy of diagonal line lengths | Drops (spectrum concentrates) |
| **LAM** (Laminarity) | Fraction in vertical lines | Rises (trap states) |
| **TT** (Trapping Time) | Mean vertical line length | Rises (longer in same state) |

## Quick Start

```python
from rqa_factor import compute_rqa_factors

factors = compute_rqa_factors(returns, window=63, dim=3, tau=1)
# factors['det'], factors['rr'], factors['lmax'], etc.
```

## Use

```bash
# test
cd tests && python test_rqa.py

# demo
cd notebooks && python demo_rqa.py
```

## How It Works

1. **Embedding**: Daily return → phase-space vectors via Takens' theorem
2. **Recurrence**: Points within `eps` distance → binary matrix
3. **Measures**: Diagonal/vertical line statistics from the matrix

## ⚠️ Practice Statement

Practice exercise in quant factor mining. Synthetic data only — not validated on live markets. Not investment advice.

## References

- Zbilut, J.P. & Webber, C.L. (1992). "Embeddings and delays as derived from quantification of recurrence plots." *Physics Letters A*, 171(3-4).
- Marwan, N. et al. (2007). "Recurrence plots for the analysis of complex systems." *Physics Reports*, 438(5-6).
- Webber, C.L. & Zbilut, J.P. (1994). "Dynamical assessment of physiological systems and states using recurrence plot strategies." *Journal of Applied Physiology*, 76(2).

## License

MIT
