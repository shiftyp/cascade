"""Unit Tests: IQ Trajectory Generation"""

import pytest
import sys
import numpy as np

sys.path.insert(0, '/workspaces/cascade')

from modules.training.patterns.iq_trajectories import generate_iq_trajectory
from modules.training.patterns.optimizer import compute_iq_complexity, mutate_iq_directly


def test_emergency_patterns_produce_bpsk():
    """T025: Test λ=0.0 produces BPSK line on I-axis"""
    iq = generate_iq_trajectory(lambda_complexity=0.0, seed=42)

    # Should be all real, positive (BPSK on I-axis)
    assert iq.shape == (32,)
    assert iq.dtype == np.dtype('complex64')

    # Check that imaginary part is zero (or very small)
    assert np.allclose(iq.imag, 0, atol=1e-6), "BPSK should have zero imaginary component"

    # Check that real part is constant
    real_std = np.std(iq.real)
    assert real_std < 0.1, f"BPSK should have constant magnitude, got std={real_std}"


def test_simple_patterns_produce_circles():
    """T025: Test λ=0.1-0.3 produces circular IQ patterns"""
    for lam in [0.1, 0.2, 0.3]:
        iq = generate_iq_trajectory(lambda_complexity=lam, seed=42)

        # Check magnitude is roughly constant (circular)
        # Note: λ=0.3 is boundary to ellipses, so allow higher std
        magnitudes = np.abs(iq)
        mag_std = np.std(magnitudes)
        max_std = 0.2 if lam < 0.3 else 0.35
        assert mag_std < max_std, f"Circles should have constant magnitude, got std={mag_std} for λ={lam}"

        # Check phase varies (not BPSK)
        phases = np.angle(iq)
        phase_range = np.ptp(phases)
        assert phase_range > 1.0, f"Circles should have varying phase, got range={phase_range} for λ={lam}"


def test_complex_patterns_produce_lissajous():
    """T025: Test λ=0.7-0.9 produces complex Lissajous patterns"""
    for lam in [0.7, 0.8, 0.9]:
        iq = generate_iq_trajectory(lambda_complexity=lam, seed=42)

        # Check both magnitude AND phase vary (Lissajous)
        magnitudes = np.abs(iq)
        mag_std = np.std(magnitudes)
        assert mag_std > 0.1, f"Lissajous should have varying magnitude, got std={mag_std} for λ={lam}"

        phases = np.angle(iq)
        phase_range = np.ptp(phases)
        assert phase_range > 2.0, f"Lissajous should have large phase variation, got range={phase_range} for λ={lam}"


def test_iq_trajectory_normalized():
    """T025: Test IQ trajectories have unit power"""
    for lam in [0.0, 0.2, 0.5, 0.9]:
        iq = generate_iq_trajectory(lambda_complexity=lam, seed=42)

        power = np.mean(np.abs(iq) ** 2)
        assert abs(power - 1.0) < 0.01, f"IQ should have unit power, got {power} for λ={lam}"


def test_iq_complexity_measurement():
    """T025: Test compute_iq_complexity() returns valid λ"""
    # BPSK (should measure as λ≈0)
    bpsk_iq = np.ones(32, dtype='complex64')
    lam = compute_iq_complexity(bpsk_iq)
    assert lam < 0.1, f"BPSK should measure low complexity, got λ={lam}"

    # Random complex (should measure higher)
    random_iq = (np.random.randn(32) + 1j * np.random.randn(32)).astype('complex64')
    random_iq = random_iq / np.sqrt(np.mean(np.abs(random_iq)**2))
    lam = compute_iq_complexity(random_iq)
    assert lam > 0.3, f"Random IQ should measure high complexity, got λ={lam}"
    assert lam <= 1.0, f"λ should be <= 1.0, got {lam}"


def test_iq_mutation_preserves_normalization():
    """T025: Test mutate_iq_directly() maintains unit power"""
    iq = np.ones(32, dtype='complex64')

    for _ in range(10):
        iq = mutate_iq_directly(iq, noise_scale=0.2, rng=np.random.RandomState(42))

        power = np.mean(np.abs(iq) ** 2)
        assert abs(power - 1.0) < 0.01, f"Mutated IQ should maintain unit power, got {power}"


def test_deterministic_with_seed():
    """T025: Test IQ generation is deterministic with seed"""
    iq1 = generate_iq_trajectory(lambda_complexity=0.5, seed=42)
    iq2 = generate_iq_trajectory(lambda_complexity=0.5, seed=42)

    assert np.allclose(iq1, iq2), "Same seed should produce same IQ trajectory"
