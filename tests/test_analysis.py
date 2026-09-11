import numpy as np

from src.analysis import structural_summary
from src.config import get_config
from src.dynamics import linearize_system


def test_linearized_system_is_stable_controllable_observable():
    cfg = get_config()
    A, B, C = linearize_system(cfg)
    summary = structural_summary(A, B, C)

    assert np.all(np.real(summary["eigenvalues"]) < 0)
    assert summary["controllability_rank"] == 2
    assert summary["observability_rank"] == 2
    assert summary["controllable"]
    assert summary["observable"]


def test_unstable_system_flagged_correctly():
    A = np.array([[0.0, 1.0], [5.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    summary = structural_summary(A, B, C)
    assert np.any(np.real(summary["eigenvalues"]) > 0)
