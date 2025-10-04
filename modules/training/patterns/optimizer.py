"""Simulated Annealing Optimizer for Pattern Orthogonality

UPDATED 2025-10-04: Supports two optimization modes:
1. Direct IQ mutation: Mutates IQ points directly in complex plane
2. Lambda-based: Mutates λ parameter to select from IQ shape families

Primary objective: -37.5 dB orthogonality
Secondary objective: Minimize IQ complexity
"""

from typing import List, Tuple, Optional
import numpy as np
from .models import Pattern
from .correlation import compute_4d_correlation
from .iq_trajectories import generate_iq_trajectory


def optimize_pattern(
    pattern_id: int,
    base_freq_sequence: np.ndarray,
    existing_patterns: List[Pattern],
    target_db: float = -37.5,
    max_iterations: int = 100000,
    seed: int = None
) -> Tuple[np.ndarray, float]:
    """Optimize pattern using simulated annealing

    Optimizes BOTH:
    1. Tone sequence (freq_sequence)
    2. IQ complexity (λ parameter)

    Cost function balances:
    - Primary: Achieve orthogonality (<-37.5 dB vs all existing patterns)
    - Secondary: Minimize λ (prefer simpler IQ trajectories)

    Args:
        pattern_id: ID for this pattern
        base_freq_sequence: Initial frequency sequence
        existing_patterns: Already generated patterns to avoid
        target_db: Target correlation threshold in dB
        max_iterations: Maximum optimization iterations
        seed: Random seed for reproducibility

    Returns:
        (optimized_freq_sequence, optimized_lambda)
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    # Start with base pattern and λ=0.0 (simplest)
    current_freq = base_freq_sequence.copy()
    current_lambda = 0.0

    # Generate initial IQ trajectory
    current_iq = generate_iq_trajectory(current_lambda, seed=seed)

    # Create Pattern object for correlation calculation
    current_pattern = Pattern(
        pattern_id=pattern_id,
        freq_sequence=current_freq,
        iq_trajectory=current_iq,
        iq_complexity_lambda=current_lambda
    )

    # Compute initial cost
    current_cost = _compute_cost(current_pattern, existing_patterns, target_db)

    # Best solution found so far
    best_freq = current_freq.copy()
    best_lambda = current_lambda
    best_cost = current_cost

    # Simulated annealing parameters
    temperature = 10.0
    cooling_rate = 0.9999

    for iteration in range(max_iterations):
        # Try mutation
        if rng.random() < 0.7:
            # Mutate tone sequence (70% of mutations)
            new_freq = current_freq.copy()
            mut_index = rng.randint(0, 32)
            new_freq[mut_index] = rng.randint(0, 4)
            new_lambda = current_lambda
        else:
            # Mutate λ (30% of mutations)
            new_freq = current_freq.copy()
            # Small random walk in λ
            delta_lambda = rng.uniform(-0.05, 0.05)
            new_lambda = np.clip(current_lambda + delta_lambda, 0.0, 0.9)

        # Generate new IQ trajectory
        new_iq = generate_iq_trajectory(new_lambda, seed=seed + iteration if seed else None)

        # Create new pattern
        new_pattern = Pattern(
            pattern_id=pattern_id,
            freq_sequence=new_freq,
            iq_trajectory=new_iq,
            iq_complexity_lambda=new_lambda
        )

        # Compute cost
        new_cost = _compute_cost(new_pattern, existing_patterns, target_db)

        # Accept or reject
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / temperature):
            current_freq = new_freq
            current_lambda = new_lambda
            current_pattern = new_pattern
            current_cost = new_cost

            # Update best if improved
            if current_cost < best_cost:
                best_freq = current_freq.copy()
                best_lambda = current_lambda
                best_cost = current_cost

        # Cool temperature
        temperature *= cooling_rate

        # Early stopping if achieved target with minimal λ
        if best_cost < 0.1:  # Good enough (orthogonal + low λ)
            break

    return best_freq, best_lambda


def _compute_cost(
    pattern: Pattern,
    existing_patterns: List[Pattern],
    target_db: float
) -> float:
    """Compute cost function for optimization

    Cost = orthogonality_violation + lambda_penalty

    Args:
        pattern: Pattern to evaluate
        existing_patterns: Patterns to check against
        target_db: Target correlation threshold

    Returns:
        Cost value (lower is better, 0 is perfect)
    """
    if len(existing_patterns) == 0:
        # No existing patterns to compare against
        # Only penalize λ
        return pattern.iq_complexity_lambda

    # Find worst-case correlation
    max_correlation_db = -float('inf')
    for existing in existing_patterns:
        corr_db = compute_4d_correlation(pattern, existing)
        max_correlation_db = max(max_correlation_db, corr_db)

    # Orthogonality violation (primary objective)
    # Positive if violates target, 0 if meets target
    orthogonality_violation = max(0, max_correlation_db - target_db)

    # Lambda penalty (secondary objective)
    # Prefer lower λ (simpler IQ)
    lambda_penalty = pattern.iq_complexity_lambda * 0.1  # Weight λ at 10% of orthogonality

    # Total cost
    cost = orthogonality_violation + lambda_penalty

    return cost


def mutate_iq_directly(
    iq_trajectory: np.ndarray,
    noise_scale: float = 0.1,
    rng: np.random.RandomState = None
) -> np.ndarray:
    """Mutate IQ trajectory by adding complex noise

    Args:
        iq_trajectory: Current IQ trajectory (32 × complex64)
        noise_scale: Standard deviation of noise to add
        rng: Random number generator

    Returns:
        Mutated IQ trajectory
    """
    if rng is None:
        rng = np.random

    # Add complex Gaussian noise
    real_noise = rng.normal(0, noise_scale, size=32)
    imag_noise = rng.normal(0, noise_scale, size=32)
    noise = real_noise + 1j * imag_noise

    new_iq = iq_trajectory + noise

    # Normalize to unit power
    power = np.mean(np.abs(new_iq) ** 2)
    new_iq = new_iq / np.sqrt(power)

    return new_iq.astype('complex64')


def compute_iq_complexity(iq_trajectory: np.ndarray) -> float:
    """Compute IQ complexity metric (λ) from trajectory

    Measures how much the IQ pattern deviates from simple BPSK.
    Higher complexity = more use of IQ constellation space.

    Args:
        iq_trajectory: IQ trajectory to measure (32 × complex64)

    Returns:
        Complexity λ in range [0.0, 1.0]
    """
    # Measure 1: Magnitude variation (0 = constant magnitude, 1 = high variation)
    magnitudes = np.abs(iq_trajectory)
    mag_std = np.std(magnitudes)
    mag_complexity = np.clip(mag_std * 2, 0, 1)  # Scale to [0,1]

    # Measure 2: Phase variation (0 = constant phase, 1 = full rotation)
    phases = np.angle(iq_trajectory)
    phase_diffs = np.diff(phases)
    # Wrap phase differences to [-π, π]
    phase_diffs = np.arctan2(np.sin(phase_diffs), np.cos(phase_diffs))
    phase_complexity = np.clip(np.std(phase_diffs) / np.pi, 0, 1)

    # Combined complexity (average of both measures)
    lambda_complexity = (mag_complexity + phase_complexity) / 2

    return float(np.clip(lambda_complexity, 0.0, 0.9))


def optimize_pattern_direct_iq(
    pattern_id: int,
    base_freq_sequence: np.ndarray,
    existing_patterns: List[Pattern],
    target_db: float = -37.5,
    max_iterations: int = 100000,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Optimize pattern using direct IQ mutation (not λ-based)

    This is the new adaptive approach: mutates IQ points directly
    rather than selecting from predefined shapes.

    Args:
        pattern_id: ID for this pattern
        base_freq_sequence: Initial frequency sequence
        existing_patterns: Already generated patterns
        target_db: Target correlation threshold
        max_iterations: Maximum iterations
        seed: Random seed

    Returns:
        (optimized_freq_sequence, optimized_iq_trajectory, measured_lambda)
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    # Start with BPSK (simplest IQ)
    current_freq = base_freq_sequence.copy()
    current_iq = np.ones(32, dtype='complex64')  # All 1+0j

    # Create pattern
    current_pattern = Pattern(
        pattern_id=pattern_id,
        freq_sequence=current_freq,
        iq_trajectory=current_iq,
        iq_complexity_lambda=0.0
    )

    current_cost = _compute_cost(current_pattern, existing_patterns, target_db)

    # Best so far
    best_freq = current_freq.copy()
    best_iq = current_iq.copy()
    best_cost = current_cost

    # SA parameters
    temperature = 10.0
    cooling_rate = 0.9999
    noise_scale = 0.3  # Initial IQ mutation strength

    for iteration in range(max_iterations):
        # Decide what to mutate
        mutation_type = rng.random()

        if mutation_type < 0.5:
            # Mutate frequency sequence (50%)
            new_freq = current_freq.copy()
            mut_index = rng.randint(0, 32)
            new_freq[mut_index] = rng.randint(0, 4)
            new_iq = current_iq.copy()

        else:
            # Mutate IQ trajectory directly (50%)
            new_freq = current_freq.copy()
            new_iq = mutate_iq_directly(current_iq, noise_scale, rng)

        # Measure complexity of new IQ
        new_lambda = compute_iq_complexity(new_iq)

        # Create new pattern
        new_pattern = Pattern(
            pattern_id=pattern_id,
            freq_sequence=new_freq,
            iq_trajectory=new_iq,
            iq_complexity_lambda=new_lambda
        )

        # Compute cost
        new_cost = _compute_cost(new_pattern, existing_patterns, target_db)

        # Accept or reject
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / temperature):
            current_freq = new_freq
            current_iq = new_iq
            current_pattern = new_pattern
            current_cost = new_cost

            if current_cost < best_cost:
                best_freq = current_freq.copy()
                best_iq = current_iq.copy()
                best_cost = current_cost

        # Cool temperature
        temperature *= cooling_rate

        # Adapt noise scale based on progress
        if iteration % 10000 == 0 and iteration > 0:
            noise_scale *= 0.9  # Decrease mutation strength over time

        # Early stopping
        if best_cost < 0.05:
            break

    # Measure final complexity
    final_lambda = compute_iq_complexity(best_iq)

    return best_freq, best_iq, final_lambda


def adaptive_lambda_search(
    pattern_id: int,
    base_freq_sequence: np.ndarray,
    existing_patterns: List[Pattern],
    target_db: float = -37.5,
    seed: int = None
) -> Tuple[np.ndarray, float]:
    """Find minimum λ needed to achieve orthogonality

    Incrementally increases λ until target separation achieved.

    Args:
        pattern_id: ID for this pattern
        base_freq_sequence: Initial frequency sequence
        existing_patterns: Existing patterns
        target_db: Target correlation threshold
        seed: Random seed

    Returns:
        (optimized_freq_sequence, minimum_lambda)
    """
    lambda_candidates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                        0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]

    for max_lambda in lambda_candidates:
        # Try optimizing with this λ constraint
        freq, lam = optimize_pattern(
            pattern_id=pattern_id,
            base_freq_sequence=base_freq_sequence,
            existing_patterns=existing_patterns,
            target_db=target_db,
            max_iterations=50000,  # Shorter iterations for search
            seed=seed
        )

        # Generate IQ for validation
        iq = generate_iq_trajectory(lam, seed=seed)

        # Create pattern and check correlation
        pattern = Pattern(
            pattern_id=pattern_id,
            freq_sequence=freq,
            iq_trajectory=iq,
            iq_complexity_lambda=lam
        )

        # Check if achieved target
        if len(existing_patterns) > 0:
            max_corr = max([compute_4d_correlation(pattern, existing)
                           for existing in existing_patterns])

            if max_corr < target_db:
                # Success! This λ is sufficient
                return freq, lam

    # If we get here, even λ=0.9 wasn't enough (unlikely)
    # Return best effort
    return freq, lambda_candidates[-1]


def _compute_cost_phase_aware(
    pattern: Pattern,
    existing_patterns: List[Pattern],
    target_db: float,
    num_phase_samples: int = 5
) -> float:
    """Compute cost function with phase distortion robustness

    Tests correlation under random phase scenarios during optimization
    to ensure patterns are robust to HF propagation effects.

    Args:
        pattern: Pattern to evaluate
        existing_patterns: Patterns to check against
        target_db: Target correlation threshold
        num_phase_samples: Number of random phase scenarios to test

    Returns:
        Cost value (worst-case across phase scenarios)
    """
    if len(existing_patterns) == 0:
        return pattern.iq_complexity_lambda

    # Find worst-case correlation across phase scenarios
    worst_correlation_db = -float('inf')

    for existing in existing_patterns:
        # Sample multiple phase scenarios
        for _ in range(num_phase_samples):
            # Random phase per tone (models frequency-dependent distortion)
            phase_per_tone = np.random.uniform(-np.pi, np.pi, size=4)
            # Random phase per symbol (models time-varying channel)
            phase_per_symbol = np.random.uniform(-0.2, 0.2, size=32)

            from .correlation import compute_correlation_with_phase
            corr_db = compute_correlation_with_phase(
                pattern,
                existing,
                phase_per_tone,
                phase_per_symbol
            )

            worst_correlation_db = max(worst_correlation_db, corr_db)

    # Orthogonality violation (using worst-case)
    orthogonality_violation = max(0, worst_correlation_db - target_db)

    # Lambda penalty
    lambda_penalty = pattern.iq_complexity_lambda * 0.1

    # Total cost
    cost = orthogonality_violation + lambda_penalty

    return cost


def optimize_pattern_two_phase(
    pattern_id: int,
    base_freq_sequence: np.ndarray,
    existing_patterns: List[Pattern],
    target_db: float = -37.5,
    freq_iterations: int = 150000,
    iq_iterations: int = 50000,
    phase_aware: bool = True,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Two-phase optimization: frequency first (λ=0), then add IQ if needed

    This approach tries to maximize patterns that can use BPSK (λ=0) before
    adding IQ complexity, leading to overall lower average λ.

    Args:
        pattern_id: ID for this pattern
        base_freq_sequence: Initial frequency sequence
        existing_patterns: Already generated patterns
        target_db: Target correlation threshold
        freq_iterations: Iterations for frequency-only phase
        iq_iterations: Iterations for IQ refinement phase
        phase_aware: Use phase-aware cost function
        seed: Random seed

    Returns:
        (optimized_freq_sequence, optimized_iq_trajectory, measured_lambda)
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    # === PHASE 1: Frequency-Only Optimization (λ=0) ===
    print(f"    Phase 1: Frequency-only ({freq_iterations:,} iterations)...", end=' ', flush=True)

    current_freq = base_freq_sequence.copy()
    current_iq = np.ones(32, dtype='complex64')  # BPSK

    current_pattern = Pattern(
        pattern_id=pattern_id,
        freq_sequence=current_freq,
        iq_trajectory=current_iq,
        iq_complexity_lambda=0.0
    )

    # Choose cost function
    if phase_aware:
        current_cost = _compute_cost_phase_aware(current_pattern, existing_patterns, target_db, num_phase_samples=3)
    else:
        current_cost = _compute_cost(current_pattern, existing_patterns, target_db)

    best_freq = current_freq.copy()
    best_cost = current_cost

    # SA parameters for frequency-only phase
    temperature = 10.0
    cooling_rate = 0.99998  # Slower cooling for longer run

    for iteration in range(freq_iterations):
        # Mutate frequency only
        new_freq = current_freq.copy()
        mut_index = rng.randint(0, 32)
        new_freq[mut_index] = rng.randint(0, 4)

        new_pattern = Pattern(
            pattern_id=pattern_id,
            freq_sequence=new_freq,
            iq_trajectory=current_iq,  # Still BPSK
            iq_complexity_lambda=0.0
        )

        # Compute cost
        if phase_aware:
            new_cost = _compute_cost_phase_aware(new_pattern, existing_patterns, target_db, num_phase_samples=3)
        else:
            new_cost = _compute_cost(new_pattern, existing_patterns, target_db)

        # Accept or reject
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / temperature):
            current_freq = new_freq
            current_pattern = new_pattern
            current_cost = new_cost

            if current_cost < best_cost:
                best_freq = current_freq.copy()
                best_cost = current_cost

        temperature *= cooling_rate

    # Check if we achieved target with BPSK
    phase1_pattern = Pattern(pattern_id, best_freq, current_iq, 0.0)

    if phase_aware:
        final_corr = max([
            max([compute_4d_correlation(phase1_pattern, ex) for _ in range(10)])
            for ex in existing_patterns
        ]) if existing_patterns else -100
    else:
        final_corr = max([compute_4d_correlation(phase1_pattern, ex) for ex in existing_patterns]) if existing_patterns else -100

    if final_corr < target_db:
        # Success with BPSK!
        print(f"λ=0.000 (BPSK sufficient!)")
        return best_freq, current_iq, 0.0

    print(f"λ=0 insufficient ({final_corr:.1f} dB), adding IQ...")

    # === PHASE 2: IQ Refinement ===
    print(f"    Phase 2: IQ refinement ({iq_iterations:,} iterations)...", end=' ', flush=True)

    current_freq = best_freq.copy()  # Start from best frequency
    current_iq = np.ones(32, dtype='complex64')  # Start with BPSK
    current_lambda = 0.0

    current_pattern = Pattern(pattern_id, current_freq, current_iq, current_lambda)
    current_cost = _compute_cost_phase_aware(current_pattern, existing_patterns, target_db, 3) if phase_aware else _compute_cost(current_pattern, existing_patterns, target_db)

    best_iq = current_iq.copy()
    best_lambda = 0.0
    best_cost_phase2 = current_cost

    temperature = 5.0  # Lower initial temp for refinement
    cooling_rate = 0.99995
    noise_scale = 0.2

    for iteration in range(iq_iterations):
        # Mutate both frequency (30%) and IQ (70%)
        if rng.random() < 0.3:
            # Occasional frequency adjustment
            new_freq = current_freq.copy()
            mut_index = rng.randint(0, 32)
            new_freq[mut_index] = rng.randint(0, 4)
            new_iq = current_iq.copy()
        else:
            # IQ mutation (primary)
            new_freq = current_freq.copy()
            new_iq = mutate_iq_directly(current_iq, noise_scale, rng)

        new_lambda = compute_iq_complexity(new_iq)

        new_pattern = Pattern(pattern_id, new_freq, new_iq, new_lambda)

        if phase_aware:
            new_cost = _compute_cost_phase_aware(new_pattern, existing_patterns, target_db, 3)
        else:
            new_cost = _compute_cost(new_pattern, existing_patterns, target_db)

        # Accept or reject
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / temperature):
            current_freq = new_freq
            current_iq = new_iq
            current_lambda = new_lambda
            current_pattern = new_pattern
            current_cost = new_cost

            if current_cost < best_cost_phase2:
                best_freq = current_freq.copy()
                best_iq = current_iq.copy()
                best_lambda = new_lambda
                best_cost_phase2 = current_cost

        temperature *= cooling_rate

        # Adapt noise
        if iteration % 5000 == 0 and iteration > 0:
            noise_scale *= 0.95

        # Early stopping
        if best_cost_phase2 < 0.05:
            break

    final_lambda = compute_iq_complexity(best_iq)
    print(f"λ={final_lambda:.3f}")

    return best_freq, best_iq, final_lambda
