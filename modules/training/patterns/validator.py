"""Pattern Validation and Orthogonality Checking"""

from typing import List, Tuple
import numpy as np
from .models import Pattern
from .correlation import compute_4d_correlation, compute_robust_correlation
from .binary_format import load_pattern_file


def validate_orthogonality(patterns: List[Pattern], target_db: float = -37.5) -> Tuple[bool, dict]:
    """Validate that all pattern pairs meet orthogonality threshold

    Args:
        patterns: List of patterns to validate
        target_db: Required correlation threshold (default -37.5 dB)

    Returns:
        (pass/fail, statistics dict with min/max/mean correlation and violations)
    """
    n = len(patterns)
    correlations = []
    failed_pairs = []

    # Check all pairs
    for i in range(n):
        for j in range(i + 1, n):
            corr_db = compute_4d_correlation(patterns[i], patterns[j])
            correlations.append(corr_db)

            if corr_db > target_db:
                failed_pairs.append((patterns[i].pattern_id, patterns[j].pattern_id, corr_db))

    # Compute statistics
    if len(correlations) > 0:
        min_corr = float(np.min(correlations))
        max_corr = float(np.max(correlations))
        mean_corr = float(np.mean(correlations))
    else:
        min_corr = max_corr = mean_corr = 0.0

    stats = {
        'pair_count': len(correlations),
        'min_correlation_db': min_corr,
        'max_correlation_db': max_corr,
        'mean_correlation_db': mean_corr,
        'failed_pairs': failed_pairs,
        'target_db': target_db
    }

    passes = len(failed_pairs) == 0

    return passes, stats


def generate_validation_report(pattern_file: str) -> str:
    """Generate markdown validation report for pattern file

    Args:
        pattern_file: Path to pattern file

    Returns:
        Markdown-formatted validation report
    """
    # Load patterns
    patterns = load_pattern_file(pattern_file)

    # Validate
    passes, stats = validate_orthogonality(patterns, target_db=-37.5)

    # Generate report
    report = f"""# CASCADE Pattern Validation Report

**File**: `{pattern_file}`
**Pattern Count**: {len(patterns)}
**Target Threshold**: {stats['target_db']} dB

## Orthogonality Results

**Status**: {'✓ PASS' if passes else '✗ FAIL'}

**Statistics**:
- Pattern pairs checked: {stats['pair_count']}
- Min correlation: {stats['min_correlation_db']:.2f} dB
- Max correlation: {stats['max_correlation_db']:.2f} dB
- Mean correlation: {stats['mean_correlation_db']:.2f} dB

"""

    if stats['failed_pairs']:
        report += f"""## Failed Pairs ({len(stats['failed_pairs'])})

| Pattern A | Pattern B | Correlation (dB) |
|-----------|-----------|------------------|
"""
        for pid_i, pid_j, corr in stats['failed_pairs'][:20]:  # Show first 20
            report += f"| {pid_i} | {pid_j} | {corr:.2f} |\n"

        if len(stats['failed_pairs']) > 20:
            report += f"\n... and {len(stats['failed_pairs']) - 20} more failures\n"
    else:
        report += "## All pairs pass orthogonality threshold ✓\n"

    # Lambda statistics
    lambdas = [p.iq_complexity_lambda for p in patterns]
    report += f"""
## IQ Complexity (λ) Distribution

- Min λ: {min(lambdas):.3f}
- Max λ: {max(lambdas):.3f}
- Mean λ: {np.mean(lambdas):.3f}
- Median λ: {np.median(lambdas):.3f}

**Note**: Lower λ = simpler IQ trajectories = better robustness
"""

    return report


def test_phase_robustness(
    patterns: List[Pattern],
    sample_size: int = 100,
    num_phase_trials: int = 100
) -> dict:
    """Test pattern orthogonality under phase distortion

    Samples a subset of pattern pairs and tests correlation under random
    phase rotation to simulate HF propagation effects.

    Args:
        patterns: List of patterns to test
        sample_size: Number of random pairs to test
        num_phase_trials: Phase scenarios per pair

    Returns:
        Dict with phase robustness statistics
    """
    n = len(patterns)

    # Sample random pairs
    num_pairs = n * (n - 1) // 2
    if num_pairs > sample_size:
        # Random sampling
        pairs_to_test = []
        rng = np.random.RandomState(42)
        while len(pairs_to_test) < sample_size:
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i != j and (i, j) not in pairs_to_test and (j, i) not in pairs_to_test:
                pairs_to_test.append((min(i, j), max(i, j)))
    else:
        # Test all pairs
        pairs_to_test = [(i, j) for i in range(n) for j in range(i + 1, n)]

    ideal_correlations = []
    robust_correlations = []

    print(f"Testing phase robustness on {len(pairs_to_test)} pairs...")

    for i, j in pairs_to_test:
        # Ideal correlation
        ideal_corr = compute_4d_correlation(patterns[i], patterns[j])
        ideal_correlations.append(ideal_corr)

        # Robust correlation (worst-case under phase distortion)
        robust_corr = compute_robust_correlation(patterns[i], patterns[j], num_phase_trials)
        robust_correlations.append(robust_corr)

    results = {
        'pairs_tested': len(pairs_to_test),
        'ideal_min_db': float(np.min(ideal_correlations)),
        'ideal_max_db': float(np.max(ideal_correlations)),
        'ideal_mean_db': float(np.mean(ideal_correlations)),
        'robust_min_db': float(np.min(robust_correlations)),
        'robust_max_db': float(np.max(robust_correlations)),
        'robust_mean_db': float(np.mean(robust_correlations)),
        'degradation_db': float(np.mean(robust_correlations) - np.mean(ideal_correlations))
    }

    return results
