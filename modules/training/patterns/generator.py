"""Pattern Set Generation Orchestrator

UPDATED 2025-10-04: Lambda minimization approach
All patterns start with λ=0.0 and optimizer increases only if needed for orthogonality.
"""

from typing import List
import numpy as np
from .models import Pattern
from .zadoff_chu import generate_zadoff_chu_pattern, generate_random_pattern
from .iq_trajectories import generate_iq_trajectory
from .optimizer import optimize_pattern_two_phase
from .validator import validate_orthogonality


def generate_pattern_set(count: int, seed: int) -> List[Pattern]:
    """Generate 64 or 128 patterns with -37.5 dB orthogonality

    New optimization strategy (2025-10-04):
    - All patterns start with λ=0.0 (BPSK, maximum robustness)
    - Simulated annealing optimizes BOTH tone sequence AND λ
    - Primary objective: Achieve -37.5 dB orthogonality
    - Secondary objective: Minimize λ (prefer simpler IQ)

    Pattern structure:
    - First 48 patterns (0-47): Beacon patterns
    - Remaining patterns: Message patterns
      - 64-set: 16 message patterns (48-63)
      - 128-set: 80 message patterns (48-127)

    Args:
        count: Number of patterns (64 or 128)
        seed: Random seed for reproducibility

    Returns:
        List of Pattern objects, all meeting -37.5 dB orthogonality

    Raises:
        ValueError: If count not in {64, 128}
    """
    if count not in {64, 128}:
        raise ValueError(f"Pattern count must be 64 or 128, got {count}")

    rng = np.random.RandomState(seed)
    patterns = []

    print(f"Generating {count} patterns with λ minimization...")
    print(f"Using seed: {seed}")

    # Phase 1: Generate 48 beacon patterns (0-47)
    print("\n=== Phase 1: Generating 48 beacon patterns ===")
    for i in range(48):
        if i < 31:
            # Use Zadoff-Chu base (excellent starting point)
            base_freq = generate_zadoff_chu_pattern(u=i)
            source = "Zadoff-Chu"
        else:
            # Random initialization for remaining beacon patterns
            base_freq = generate_random_pattern(seed + i)
            source = "Random"

        # Use two-phase optimization with phase-aware cost function
        # Default: 80% freq-only, 20% IQ refinement
        total_iterations = 200000  # Beacon patterns use half budget
        freq_iter = int(total_iterations * 0.8)
        iq_iter = int(total_iterations * 0.2)

        optimized_freq, optimized_iq, optimized_lambda = optimize_pattern_two_phase(
            pattern_id=i,
            base_freq_sequence=base_freq,
            existing_patterns=patterns,
            target_db=-37.5,
            freq_iterations=freq_iter,
            iq_iterations=iq_iter,
            phase_aware=True,
            seed=seed + i
        )

        pattern = Pattern(
            pattern_id=i,
            freq_sequence=optimized_freq,
            iq_trajectory=optimized_iq,
            iq_complexity_lambda=optimized_lambda
        )

        patterns.append(pattern)
        print(f"  Pattern {i:3d}: {source:12s} → λ={optimized_lambda:.3f}")

    # Phase 2: Generate message patterns
    num_message = count - 48
    print(f"\n=== Phase 2: Generating {num_message} message patterns ===")

    for i in range(48, count):
        # Random initialization
        base_freq = generate_random_pattern(seed + i)

        # Use two-phase optimization (full budget for message patterns)
        total_iterations = 400000  # Message patterns use full budget
        freq_iter = int(total_iterations * 0.8)
        iq_iter = int(total_iterations * 0.2)

        optimized_freq, optimized_iq, optimized_lambda = optimize_pattern_two_phase(
            pattern_id=i,
            base_freq_sequence=base_freq,
            existing_patterns=patterns,
            target_db=-37.5,
            freq_iterations=freq_iter,
            iq_iterations=iq_iter,
            phase_aware=True,
            seed=seed + i
        )

        pattern = Pattern(
            pattern_id=i,
            freq_sequence=optimized_freq,
            iq_trajectory=optimized_iq,
            iq_complexity_lambda=optimized_lambda
        )

        patterns.append(pattern)
        print(f"  Pattern {i:3d}: Message      → λ={optimized_lambda:.3f}")

    # Phase 3: Final validation
    print("\n=== Phase 3: Final validation ===")
    passes, stats = validate_orthogonality(patterns, target_db=-37.5)

    print(f"Pair count: {stats['pair_count']}")
    print(f"Min correlation: {stats['min_correlation_db']:.2f} dB")
    print(f"Max correlation: {stats['max_correlation_db']:.2f} dB")
    print(f"Mean correlation: {stats['mean_correlation_db']:.2f} dB")

    # Lambda statistics
    lambdas = [p.iq_complexity_lambda for p in patterns]
    bpsk_count = sum(1 for lam in lambdas if lam < 0.05)

    print(f"\nλ distribution:")
    print(f"  Min: {min(lambdas):.3f}")
    print(f"  Max: {max(lambdas):.3f}")
    print(f"  Mean: {np.mean(lambdas):.3f}")
    print(f"  Median: {np.median(lambdas):.3f}")
    print(f"  BPSK patterns (λ<0.05): {bpsk_count}/{len(patterns)} ({100*bpsk_count/len(patterns):.1f}%)")

    if passes:
        print("\n✓ All patterns meet -37.5 dB orthogonality threshold")
    else:
        print(f"\n✗ {len(stats['failed_pairs'])} pairs FAILED orthogonality")
        raise ValueError("Pattern generation failed orthogonality validation")

    return patterns
