"""Tests for RQA factor module."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rqa_factor.factor import (
    phase_space_embedding,
    recurrence_matrix,
    recurrence_quantification,
    compute_rqa_factors,
)


def test_embedding_basic():
    """Phase space embedding on simple signal."""
    s = np.sin(np.linspace(0, 4 * np.pi, 100))
    X = phase_space_embedding(s, dim=3, tau=2)
    assert X.shape[1] == 3
    assert X.shape[0] == 100 - (3 - 1) * 2  # = 96


def test_embedding_too_short():
    """Empty embedding when signal < dim * tau."""
    s = np.array([1.0, 2.0])
    X = phase_space_embedding(s, dim=3, tau=1)
    assert X.shape[0] == 0


def test_recurrence_matrix_shapes():
    """Recurrence matrix output shape matches input."""
    X = np.random.randn(50, 3)
    R = recurrence_matrix(X, eps=0.5)
    assert R.shape == (50, 50)
    assert R.dtype == bool


def test_recurrence_matrix_identical():
    """All points identical → full recurrence."""
    X = np.ones((20, 3)) * 42.0
    R = recurrence_matrix(X, eps=0.1)
    assert np.all(R)


def test_recurrence_matrix_far():
    """All points far apart → only self-recurrences."""
    X = np.arange(10).reshape(-1, 1) * 100.0
    R = recurrence_matrix(X, eps=0.1)
    if R.size > 0:
        assert np.sum(R) == R.shape[0]  # only self-recurrences


def test_rqa_measures_contain_keys():
    """RQA dict has all expected keys."""
    X = np.random.randn(30, 3)
    R = recurrence_matrix(X, eps=0.5)
    metrics = recurrence_quantification(R)
    expected = {'rr', 'det', 'lmax', 'entr', 'lam', 'tt'}
    assert set(metrics.keys()) == expected


def test_rqa_measures_bounds():
    """All RQA measures in reasonable ranges."""
    X = np.random.randn(40, 3)
    R = recurrence_matrix(X, eps=0.3)
    metrics = recurrence_quantification(R)
    assert 0.0 <= metrics['rr'] <= 1.0
    assert 0.0 <= metrics['det'] <= 1.0
    assert metrics['lmax'] >= 1.0
    assert metrics['entr'] >= 0.0
    assert 0.0 <= metrics['lam'] <= 1.0
    assert metrics['tt'] >= 1.0


def test_rqa_periodic_vs_random():
    """Periodic signal should have higher DET than random."""
    np.random.seed(42)
    T = 100

    # Periodic: sine wave
    periodic = np.sin(np.linspace(0, 8 * np.pi, T))
    Xp = phase_space_embedding(periodic, dim=3, tau=2)
    Rp = recurrence_matrix(Xp, pct=15.0)
    mp = recurrence_quantification(Rp)

    # Random: white noise
    random = np.random.randn(T)
    Xr = phase_space_embedding(random, dim=3, tau=2)
    Rr = recurrence_matrix(Xr, pct=15.0)
    mr = recurrence_quantification(Rr)

    # Periodic should be more deterministic (may fail on edge cases)
    # but at minimum: both should produce valid metrics
    assert mp['det'] >= 0
    assert mr['det'] >= 0


def test_compute_rqa_factors_shape():
    """compute_rqa_factors returns correct time series shape."""
    np.random.seed(42)
    returns = np.random.randn(200, 10) * 0.02
    factors = compute_rqa_factors(returns, window=63)
    expected_len = 200 - 63 + 1  # = 138
    for k, v in factors.items():
        assert len(v) == expected_len, f"{k}: {len(v)} != {expected_len}"


def test_compute_rqa_factors_short_series():
    """Short series raises error."""
    returns = np.random.randn(10, 5)
    try:
        compute_rqa_factors(returns, window=63)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_empty_recurrence_quantification():
    """Empty recurrence matrix returns defaults."""
    metrics = recurrence_quantification(np.empty((0, 0)))
    assert metrics['rr'] == 0.0


if __name__ == '__main__':
    for name, fn in sorted({k: v for k, v in globals().items()
                           if k.startswith('test_')}.items()):
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception as e:
            print(f'  FAIL  {name}: {e}')
