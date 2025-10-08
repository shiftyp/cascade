"""Collision scenario generator for CASCADE V2.

Simulates collisions between multiple CASCADE signals for training the
neural network to handle overlapping transmissions and near-far problems.

Collision types:
- Full collision: Same pattern + frequency + time
- Pattern collision: Same pattern, different frequencies
- Frequency collision: Different patterns, same frequency
- Partial overlap: Signals offset in time

Source: CASCADE V2 protocol - RTS/CTS collision avoidance training
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CollisionSignal:
    """Single signal in collision scenario.

    Attributes:
        iq_samples: IQ signal samples (complex64)
        pattern_id: Pattern ID (0-7)
        frequency_pair: Frequency pair (0-66)
        modulation: Modulation scheme
        start_offset: Start time offset in samples
        power_db: Relative power in dB (0 dB = reference)
    """
    iq_samples: np.ndarray
    pattern_id: int
    frequency_pair: int
    modulation: str
    start_offset: int
    power_db: float


def create_full_collision(signal_a: np.ndarray, signal_b: np.ndarray,
                          power_ratio_db: float = 0.0,
                          seed: Optional[int] = None) -> np.ndarray:
    """Create full collision: two signals on same pattern + frequency.

    Args:
        signal_a: First CASCADE signal (complex64)
        signal_b: Second CASCADE signal (complex64)
        power_ratio_db: Power of signal_b relative to signal_a in dB (default: 0 = equal power)
        seed: Random seed

    Returns:
        np.ndarray: Collided signal (signal_a + scaled signal_b), complex64

    Note:
        This models the worst case: two users transmit simultaneously on the
        same logical channel despite RTS/CTS collision avoidance.
    """
    # Make signals same length
    min_len = min(len(signal_a), len(signal_b))
    sig_a = signal_a[:min_len]
    sig_b = signal_b[:min_len]

    # Scale signal_b by power ratio
    power_scale = 10 ** (power_ratio_db / 20)  # dB to linear amplitude
    sig_b_scaled = sig_b * power_scale

    # Add signals
    collided = sig_a + sig_b_scaled

    return collided.astype(np.complex64)


def create_partial_overlap(signal_a: np.ndarray, signal_b: np.ndarray,
                           offset_samples: int, power_ratio_db: float = 0.0) -> np.ndarray:
    """Create partial time overlap between two signals.

    Args:
        signal_a: First signal (complex64)
        signal_b: Second signal (complex64)
        offset_samples: Time offset of signal_b relative to signal_a (samples)
                       Positive = signal_b starts later
        power_ratio_db: Power of signal_b relative to signal_a in dB

    Returns:
        np.ndarray: Combined signal with partial overlap, complex64
    """
    # Determine output length
    if offset_samples >= 0:
        total_len = max(len(signal_a), offset_samples + len(signal_b))
    else:
        total_len = max(-offset_samples + len(signal_a), len(signal_b))

    # Create output buffer
    output = np.zeros(total_len, dtype=np.complex128)

    # Place signal_a
    if offset_samples >= 0:
        output[:len(signal_a)] += signal_a
    else:
        output[-offset_samples:-offset_samples + len(signal_a)] += signal_a

    # Place signal_b with power scaling
    power_scale = 10 ** (power_ratio_db / 20)
    sig_b_scaled = signal_b * power_scale

    if offset_samples >= 0:
        output[offset_samples:offset_samples + len(signal_b)] += sig_b_scaled
    else:
        output[:len(signal_b)] += sig_b_scaled

    return output.astype(np.complex64)


def create_near_far_scenario(signal_near: np.ndarray, signal_far: np.ndarray,
                             near_far_ratio_db: float = 20.0) -> np.ndarray:
    """Create near-far problem scenario.

    Models situation where one signal is much stronger than another
    (e.g., nearby vs distant transmitter).

    Args:
        signal_near: Near (strong) signal (complex64)
        signal_far: Far (weak) signal (complex64)
        near_far_ratio_db: Power ratio in dB (default: 20 dB = 10:1 amplitude)

    Returns:
        np.ndarray: Combined signal, complex64

    Note:
        Near-far problem is critical for neural network training as it must
        learn to detect weak signals in presence of strong interference.
    """
    # Near signal at 0 dB, far signal at -near_far_ratio_db
    return create_full_collision(signal_near, signal_far,
                                power_ratio_db=-near_far_ratio_db)


def create_multi_signal_collision(signals: List[CollisionSignal],
                                  total_duration_samples: Optional[int] = None) -> np.ndarray:
    """Create collision scenario with multiple CASCADE signals.

    Args:
        signals: List of CollisionSignal objects
        total_duration_samples: Total output duration (default: auto from signals)

    Returns:
        np.ndarray: Combined signal with all collisions, complex64
    """
    if not signals:
        raise ValueError("Must provide at least one signal")

    # Determine output length
    if total_duration_samples is None:
        max_end = max(sig.start_offset + len(sig.iq_samples) for sig in signals)
        total_duration_samples = max_end

    # Initialize output
    output = np.zeros(total_duration_samples, dtype=np.complex128)

    # Add each signal
    for sig in signals:
        # Power scaling
        power_scale = 10 ** (sig.power_db / 20)
        sig_scaled = sig.iq_samples * power_scale

        # Time offset
        start = sig.start_offset
        end = min(start + len(sig_scaled), total_duration_samples)
        duration = end - start

        # Add to output
        output[start:end] += sig_scaled[:duration]

    return output.astype(np.complex64)


def generate_collision_matrix(num_signals: int, collision_probability: float = 0.3,
                              seed: Optional[int] = None) -> np.ndarray:
    """Generate collision matrix showing which signals collide.

    Args:
        num_signals: Number of signals
        collision_probability: Probability of collision between any two signals
        seed: Random seed

    Returns:
        np.ndarray: Boolean matrix, shape (num_signals, num_signals)
                   matrix[i,j] = True if signals i and j collide

    Note:
        Useful for generating realistic collision scenarios where not all
        signals collide with each other.
    """
    rng = np.random.default_rng(seed)

    matrix = np.zeros((num_signals, num_signals), dtype=bool)

    for i in range(num_signals):
        for j in range(i + 1, num_signals):
            if rng.random() < collision_probability:
                matrix[i, j] = True
                matrix[j, i] = True

    return matrix


def estimate_sir(signal_of_interest: np.ndarray, interference: np.ndarray,
                overlap_start: int, overlap_duration: int) -> float:
    """Estimate Signal-to-Interference Ratio (SIR).

    Args:
        signal_of_interest: Desired signal (complex64)
        interference: Interfering signal (complex64)
        overlap_start: Start of overlap region (samples)
        overlap_duration: Duration of overlap (samples)

    Returns:
        float: SIR in dB
    """
    # Extract overlap regions
    end = overlap_start + overlap_duration
    sig_region = signal_of_interest[overlap_start:end]
    int_region = interference[overlap_start:end]

    # Calculate powers
    sig_power = np.mean(np.abs(sig_region) ** 2)
    int_power = np.mean(np.abs(int_region) ** 2)

    # Avoid log(0)
    if int_power < 1e-20:
        return np.inf

    sir_db = 10 * np.log10(sig_power / int_power)

    return sir_db


def create_collision_scenarios(signal_template: np.ndarray,
                               num_scenarios: int = 100,
                               collision_types: Optional[List[str]] = None,
                               seed: Optional[int] = None) -> List[Tuple[np.ndarray, dict]]:
    """Generate multiple collision scenarios for training.

    Args:
        signal_template: Template CASCADE signal to use
        num_scenarios: Number of scenarios to generate
        collision_types: List of collision types to include:
                        - 'full': Full collision (same time/freq/pattern)
                        - 'partial': Partial time overlap
                        - 'near_far': Near-far problem
                        - 'multi': Multiple signal collisions
        seed: Random seed

    Returns:
        List of (signal, metadata) tuples where metadata describes collision
    """
    if collision_types is None:
        collision_types = ['full', 'partial', 'near_far']

    rng = np.random.default_rng(seed)
    scenarios = []

    for scenario_idx in range(num_scenarios):
        # Choose collision type
        collision_type = rng.choice(collision_types)

        if collision_type == 'full':
            # Full collision: equal or random power ratio
            power_ratio_db = rng.uniform(-10, 10)
            collided = create_full_collision(signal_template, signal_template,
                                           power_ratio_db, rng.integers(0, 2**31))

            metadata = {
                'type': 'full_collision',
                'power_ratio_db': power_ratio_db,
                'num_signals': 2
            }

        elif collision_type == 'partial':
            # Partial overlap
            max_offset = len(signal_template) // 2
            offset = rng.integers(-max_offset, max_offset)
            power_ratio_db = rng.uniform(-5, 5)

            collided = create_partial_overlap(signal_template, signal_template,
                                             offset, power_ratio_db)

            metadata = {
                'type': 'partial_overlap',
                'offset_samples': offset,
                'power_ratio_db': power_ratio_db,
                'num_signals': 2
            }

        elif collision_type == 'near_far':
            # Near-far problem
            near_far_ratio_db = rng.uniform(10, 30)  # 10-30 dB difference

            collided = create_near_far_scenario(signal_template, signal_template,
                                               near_far_ratio_db)

            metadata = {
                'type': 'near_far',
                'near_far_ratio_db': near_far_ratio_db,
                'num_signals': 2
            }

        elif collision_type == 'multi':
            # Multiple signals
            num_signals = rng.integers(3, 6)
            signals = []

            for sig_idx in range(num_signals):
                start_offset = rng.integers(0, len(signal_template))
                power_db = rng.uniform(-10, 10)

                sig = CollisionSignal(
                    iq_samples=signal_template.copy(),
                    pattern_id=rng.integers(0, 8),
                    frequency_pair=rng.integers(0, 67),
                    modulation='QPSK',
                    start_offset=start_offset,
                    power_db=power_db
                )
                signals.append(sig)

            total_duration = int(len(signal_template) * 1.5)
            collided = create_multi_signal_collision(signals, total_duration)

            metadata = {
                'type': 'multi_signal',
                'num_signals': num_signals
            }

        else:
            continue

        scenarios.append((collided, metadata))

    return scenarios
