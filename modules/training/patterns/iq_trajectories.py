"""IQ Trajectory Generation - Adaptive Complexity

UPDATED 2025-10-04: Lambda is now minimized during optimization rather than pre-assigned.
All patterns start with λ=0.0 and optimizer increases only if needed for orthogonality.
"""

import numpy as np


def generate_iq_trajectory(lambda_complexity: float, seed: int = None) -> np.ndarray:
    """Generate IQ trajectory based on complexity level

    Complexity levels (NEW: start at λ=0.0, increase only if needed):
    - λ = 0.0: BPSK line on I-axis (maximum robustness)
    - λ = 0.1-0.3: Simple circles with radius 0.7
    - λ = 0.3-0.5: Ellipses (moderate complexity)
    - λ = 0.5-0.9: Lissajous curves (complex, used only if required)

    Args:
        lambda_complexity: IQ complexity parameter (0.0 to 0.9)
        seed: Optional random seed for reproducible trajectories

    Returns:
        32 × complex64 array representing IQ trajectory
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    t = np.linspace(0, 2 * np.pi, 32)

    if lambda_complexity < 0.1:
        # Emergency: BPSK line on I-axis (λ=0.0)
        # Maximum robustness, minimal IQ complexity
        iq_trajectory = np.ones(32, dtype='complex64')  # All (1+0j)

    elif lambda_complexity < 0.3:
        # Simple: Circles with radius 0.7
        # Phase offset for variety
        phase_offset = rng.uniform(0, 2 * np.pi) if seed is not None else 0
        iq_trajectory = 0.7 * np.exp(1j * (t + phase_offset)).astype('complex64')

    elif lambda_complexity < 0.5:
        # Moderate: Ellipses
        # Varying eccentricity based on λ
        a = 0.8  # Semi-major axis
        b = 0.3 + 0.4 * (lambda_complexity - 0.3) / 0.2  # Semi-minor axis (0.3 to 0.7)
        phase_offset = rng.uniform(0, 2 * np.pi) if seed is not None else 0
        iq_trajectory = (a * np.cos(t + phase_offset) + 1j * b * np.sin(t + phase_offset)).astype('complex64')

    else:
        # Complex: Lissajous curves (figure-8 patterns)
        # Only used if simpler IQ cannot achieve orthogonality
        freq_ratio = 1 + (lambda_complexity - 0.5) / 0.4  # 1.0 to 2.0
        phase_diff = np.pi / 4 * (lambda_complexity - 0.5) / 0.4  # 0 to π/4
        iq_trajectory = (
            np.cos(t) + 1j * np.sin(freq_ratio * t + phase_diff)
        ).astype('complex64')

    # Normalize to unit power
    power = np.mean(np.abs(iq_trajectory) ** 2)
    iq_trajectory = iq_trajectory / np.sqrt(power)

    return iq_trajectory
