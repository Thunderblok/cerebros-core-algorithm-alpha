import numpy as np

from cerebros.denseautomlstructuralcomponent.dense_automl_structural_component import (
    DenseLateralConnectivity,
)


def _sample_mean(dlc: DenseLateralConnectivity, k_minus_n: int, n: int) -> float:
    # Cast to native float for type checkers (np.mean returns numpy scalar)
    return float(np.mean([dlc.sample_lateral_connection_multiplicity(k_minus_n) for _ in range(n)]))


def test_multiplicity_mean_matches_intensity_high_lambda():
    """Empirical mean should approximate λ when decay=1 and λ>1.

    For λ=2.75 the distribution is deterministic floor(λ)=2 plus Bernoulli(frac=0.75).
    E[multiplicity]=2.75, Var=0.75*0.25=0.1875; with n=10k draws σ_mean≈0.0043.
    We allow a wide tolerance (±0.03) to avoid flakiness.
    """
    np.random.seed(1234)
    dlc = DenseLateralConnectivity(p_lateral_connection=2.75, p_lateral_connection_decay=lambda _: 1.0)
    n = 10_000
    mean_emp = _sample_mean(dlc, 0, n)
    assert abs(mean_emp - 2.75) < 0.03, f"Mean {mean_emp} deviates too much from expected 2.75"


def test_multiplicity_mean_matches_intensity_subunit_lambda():
    """λ < 1 behaves like Bernoulli(λ): mean ~ λ within tolerance."""
    np.random.seed(5678)
    dlc = DenseLateralConnectivity(p_lateral_connection=0.3, p_lateral_connection_decay=lambda _: 1.0)
    n = 10_000
    mean_emp = _sample_mean(dlc, 0, n)
    assert abs(mean_emp - 0.3) < 0.03


def test_multiplicity_mean_with_decay():
    """Mean reflects λ * decay(k). Use λ=3, decay(1)=0.5 → expected 1.5."""
    def decay(k):
        return 0.5 if k == 1 else 1.0

    np.random.seed(24680)
    dlc = DenseLateralConnectivity(p_lateral_connection=3.0, p_lateral_connection_decay=decay)
    n = 10_000
    mean_emp = _sample_mean(dlc, 1, n)
    assert abs(mean_emp - 1.5) < 0.03


def test_negative_intensity_zero_mean():
    """Negative intensity should always yield zero multiplicity (mean = 0)."""
    np.random.seed(13579)
    dlc = DenseLateralConnectivity(p_lateral_connection=-4.2)
    n = 2_000
    mean_emp = _sample_mean(dlc, 0, n)
    assert mean_emp == 0.0
