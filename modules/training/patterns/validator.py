"""Pattern Validation and Orthogonality Checking"""

from typing import List, Tuple, Dict
import numpy as np
from .models import Pattern
from .correlation import (
    compute_4d_correlation,
    compute_robust_correlation,
    compute_flip_correlation,
    compute_all_correlations,
    check_adjacent_channel_safety
)
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


def validate_flip_orthogonality(patterns: List[Pattern], target_db: float = -30.0) -> Tuple[bool, Dict]:
    """Validate flip-orthogonality for all pattern pairs

    Ensures patterns maintain sufficient separation when FSK-inverted,
    critical for adjacent-channel operation.

    Args:
        patterns: List of patterns to validate
        target_db: Target threshold for flip correlation (default -30 dB)

    Returns:
        (pass/fail, statistics dict with flip correlation metrics)
    """
    n = len(patterns)
    flip_correlations = []
    failed_flip_pairs = []

    # Check all pairs for flip-orthogonality
    for i in range(n):
        for j in range(i + 1, n):
            # Get all correlation types
            all_corrs = compute_all_correlations(patterns[i], patterns[j])

            # Check flip correlations
            flip_j = all_corrs['j_flipped']
            flip_i = all_corrs['i_flipped']
            both_flip = all_corrs['both_flipped']

            flip_correlations.extend([flip_j, flip_i, both_flip])

            # Track failures
            if flip_j > target_db:
                failed_flip_pairs.append((i, j, 'j_flipped', flip_j))
            if flip_i > target_db:
                failed_flip_pairs.append((i, j, 'i_flipped', flip_i))
            if both_flip > target_db:
                failed_flip_pairs.append((i, j, 'both_flipped', both_flip))

    # Compute statistics
    stats = {
        'pairs_checked': n * (n - 1) // 2,
        'min_flip_corr_db': float(np.min(flip_correlations)) if flip_correlations else -100.0,
        'max_flip_corr_db': float(np.max(flip_correlations)) if flip_correlations else -100.0,
        'mean_flip_corr_db': float(np.mean(flip_correlations)) if flip_correlations else -100.0,
        'failed_flip_pairs': failed_flip_pairs,
        'target_db': target_db
    }

    passes = len(failed_flip_pairs) == 0

    return passes, stats


def validate_adjacent_channel_safety(
    patterns: List[Pattern],
    tone_assignments: Dict[int, Tuple[int, int]]
) -> Dict:
    """Validate patterns for adjacent-channel operation

    Tests if patterns that might use adjacent tone pairs maintain
    sufficient orthogonality including flip-orthogonality.

    Args:
        patterns: List of patterns to validate
        tone_assignments: Mapping of pattern_id to (tone1, tone2) indices

    Returns:
        Dictionary with adjacent-channel safety metrics
    """
    results = {
        'adjacent_pairs': [],
        'safe_pairs': [],
        'unsafe_pairs': [],
        'safety_rate': 0.0
    }

    # Find patterns with adjacent tone pairs
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            pattern_i = patterns[i]
            pattern_j = patterns[j]

            # Get tone assignments
            if pattern_i.pattern_id not in tone_assignments:
                continue
            if pattern_j.pattern_id not in tone_assignments:
                continue

            tones_i = tone_assignments[pattern_i.pattern_id]
            tones_j = tone_assignments[pattern_j.pattern_id]

            # Check if adjacent (share a tone)
            if set(tones_i) & set(tones_j):
                results['adjacent_pairs'].append((i, j))

                # Check safety
                is_safe = check_adjacent_channel_safety(
                    pattern_i, pattern_j,
                    tones_i, tones_j
                )

                if is_safe:
                    results['safe_pairs'].append((i, j))
                else:
                    results['unsafe_pairs'].append((i, j))

    # Calculate safety rate
    if results['adjacent_pairs']:
        results['safety_rate'] = len(results['safe_pairs']) / len(results['adjacent_pairs'])

    return results


def generate_flip_validation_report(patterns: List[Pattern]) -> str:
    """Generate detailed flip-orthogonality validation report

    Args:
        patterns: List of patterns to validate

    Returns:
        Markdown-formatted report with flip-orthogonality analysis
    """
    # Run flip validation
    passes_flip, stats_flip = validate_flip_orthogonality(patterns, target_db=-30.0)

    # Run normal validation for comparison
    passes_normal, stats_normal = validate_orthogonality(patterns, target_db=-37.5)

    report = f"""# Flip-Orthogonality Validation Report

## Summary

**Pattern Count**: {len(patterns)}
**Normal Orthogonality**: {'✓ PASS' if passes_normal else '✗ FAIL'} (target: -37.5 dB)
**Flip Orthogonality**: {'✓ PASS' if passes_flip else '✗ FAIL'} (target: -30.0 dB)

## Normal Correlation Statistics

- Min: {stats_normal['min_correlation_db']:.2f} dB
- Max: {stats_normal['max_correlation_db']:.2f} dB
- Mean: {stats_normal['mean_correlation_db']:.2f} dB
- Failed pairs: {len(stats_normal['failed_pairs'])}

## Flip Correlation Statistics

- Min: {stats_flip['min_flip_corr_db']:.2f} dB
- Max: {stats_flip['max_flip_corr_db']:.2f} dB
- Mean: {stats_flip['mean_flip_corr_db']:.2f} dB
- Failed pairs: {len(stats_flip['failed_flip_pairs'])}

"""

    if stats_flip['failed_flip_pairs']:
        report += f"""## Failed Flip Pairs (First 20)

| Pattern A | Pattern B | Flip Type | Correlation (dB) |
|-----------|-----------|-----------|------------------|
"""
        for pid_i, pid_j, flip_type, corr in stats_flip['failed_flip_pairs'][:20]:
            report += f"| {pid_i} | {pid_j} | {flip_type} | {corr:.2f} |\n"

        if len(stats_flip['failed_flip_pairs']) > 20:
            report += f"\n... and {len(stats_flip['failed_flip_pairs']) - 20} more failures\n"
    else:
        report += "## All pairs pass flip-orthogonality threshold ✓\n"

    # Pattern-specific flip stats
    report += "\n## Per-Pattern Flip Statistics\n\n"
    report += "| Pattern | Max Flip Corr (dB) | Adjacent Safe |\n"
    report += "|---------|-------------------|---------------|\n"

    for p in patterns[:10]:  # Show first 10
        max_flip = p.flip_orthogonality_stats.get('max_flip_correlation_db', 'N/A')
        safe = p.flip_orthogonality_stats.get('adjacent_channel_safe', False)

        if isinstance(max_flip, float):
            report += f"| {p.pattern_id} | {max_flip:.2f} | {'✓' if safe else '✗'} |\n"
        else:
            report += f"| {p.pattern_id} | N/A | N/A |\n"

    report += "\n*Note: Flip-orthogonality ensures patterns remain separated when FSK-inverted due to adjacent-channel interference.*\n"

    return report
