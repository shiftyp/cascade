"""Multi-Trial Pattern Generation with Checkpointing

Runs multiple trials in parallel, saves checkpoints, and selects best result.
"""

from typing import List, Dict, Optional
from pathlib import Path
from multiprocessing import Pool
import pickle
import time
import numpy as np

from .models import Pattern
from .generator import generate_pattern_set
from .binary_format import save_pattern_file
from .validator import validate_orthogonality, test_phase_robustness
from .platform_detect import get_platform_config, optimize_for_architecture, print_platform_info
from .visualization import generate_batch_report


def run_single_trial(args: tuple) -> Dict:
    """Run a single pattern generation trial

    Args:
        args: Tuple of (trial_id, count, seed, max_iterations)

    Returns:
        Dict with trial results
    """
    trial_id, count, seed, max_iterations = args

    print(f"\n[Trial {trial_id}] Starting (seed={seed})...")

    start_time = time.time()

    try:
        # Generate patterns
        patterns = generate_pattern_set(count=count, seed=seed)

        # Validate
        passes, stats = validate_orthogonality(patterns, target_db=-37.5)

        elapsed = time.time() - start_time

        result = {
            'trial_id': trial_id,
            'seed': seed,
            'patterns': patterns,
            'passes': passes,
            'min_separation_db': stats['min_correlation_db'],
            'max_separation_db': stats['max_correlation_db'],
            'mean_separation_db': stats['mean_correlation_db'],
            'avg_lambda': float(np.mean([p.iq_complexity_lambda for p in patterns])),
            'median_lambda': float(np.median([p.iq_complexity_lambda for p in patterns])),
            'elapsed_hours': elapsed / 3600,
            'success': passes
        }

        # Compute score: balance separation and lambda
        result['score'] = result['min_separation_db'] - 0.1 * result['avg_lambda']

        print(f"[Trial {trial_id}] Complete in {result['elapsed_hours']:.2f}h - "
              f"Sep: {result['min_separation_db']:.1f} dB, λ: {result['avg_lambda']:.3f}, "
              f"Score: {result['score']:.1f}")

        return result

    except Exception as e:
        print(f"[Trial {trial_id}] FAILED: {e}")
        return {
            'trial_id': trial_id,
            'seed': seed,
            'success': False,
            'error': str(e)
        }


def generate_multi_trial(
    count: int = 128,
    num_trials: int = None,
    seed_base: int = 42,
    auto_tune: bool = True,
    max_iterations: int = None,
    checkpoint_dir: str = "modules/training/data/checkpoints",
    visualize: bool = True
) -> Dict:
    """Generate multiple trials and select best

    Optimal configuration (cost-benefit analysis):
    - Local high-end: 8 trials × 400K (depth) → -42.6 dB, λ=0.17, free
    - Cloud: 32 trials × 100K (breadth) → -40.7 dB, λ=0.22, $9.60

    Args:
        count: Number of patterns (64 or 128)
        num_trials: Number of trials (auto-detect if None, default 8)
        seed_base: Base random seed (trials use seed_base + trial_id)
        auto_tune: Auto-detect CPU and optimize
        max_iterations: Max iterations per pattern (auto-detect if None, default 400K)
        checkpoint_dir: Directory for checkpoints
        visualize: Generate visualizations

    Returns:
        Dict with best trial results
    """
    # Platform detection and auto-tuning
    if auto_tune:
        config = get_platform_config()
        print_platform_info(config)

        settings = optimize_for_architecture(config)

        if num_trials is None:
            # Depth strategy: 8 trials for deep convergence
            num_trials = min(8, settings['num_workers'])
            print(f"Auto-selected {num_trials} trials (depth strategy)")

        if max_iterations is None:
            # Prefer depth (400K) on capable hardware
            if settings['num_workers'] >= 6:
                max_iterations = 400000  # Depth strategy for high-end CPUs
                print(f"Using {max_iterations:,} iterations (depth strategy)")
            else:
                max_iterations = settings['max_iterations']  # Memory-adapted for low-end
                print(f"Using {max_iterations:,} iterations (memory-adapted)")

    else:
        if num_trials is None:
            num_trials = 8  # Changed from 16 → depth over breadth
        if max_iterations is None:
            max_iterations = 400000  # Changed from 100K → deep convergence

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Determine batch size (parallel workers)
    num_workers = min(num_trials, config['optimal_workers'] if auto_tune else 8)
    num_batches = (num_trials + num_workers - 1) // num_workers

    print(f"\nExecuting {num_trials} trials in {num_batches} batches ({num_workers} parallel workers)")
    print("=" * 60)

    all_results = []

    # Run trials in batches
    for batch_num in range(num_batches):
        batch_start = batch_num * num_workers
        batch_end = min(batch_start + num_workers, num_trials)
        batch_size = batch_end - batch_start

        print(f"\n=== Batch {batch_num + 1}/{num_batches} ({batch_size} trials) ===")

        # Prepare arguments for this batch
        trial_args = [
            (trial_id, count, seed_base + trial_id, max_iterations)
            for trial_id in range(batch_start, batch_end)
        ]

        # Run trials in parallel
        with Pool(processes=batch_size) as pool:
            batch_results = pool.map(run_single_trial, trial_args)

        all_results.extend(batch_results)

        # Save checkpoint
        checkpoint_file = checkpoint_path / f"batch_{batch_num + 1}_results.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(batch_results, f)
        print(f"\n✓ Checkpoint saved: {checkpoint_file}")

        # Find best so far
        successful = [r for r in all_results if r.get('success', False)]
        if successful:
            best_so_far = max(successful, key=lambda r: r['score'])
            print(f"\nBest so far: Trial {best_so_far['trial_id']}")
            print(f"  Score: {best_so_far['score']:.1f}")
            print(f"  Separation: {best_so_far['min_separation_db']:.1f} dB")
            print(f"  Avg λ: {best_so_far['avg_lambda']:.3f}")

            # Generate visualizations for best trial in this batch
            if visualize and 'patterns' in best_so_far:
                generate_batch_report(
                    best_so_far['patterns'],
                    batch_num + 1,
                    str(checkpoint_path.parent / "visualizations")
                )

    # Final selection
    print("\n" + "=" * 60)
    print("=== Final Selection ===")
    print("=" * 60)

    successful = [r for r in all_results if r.get('success', False)]
    if not successful:
        raise ValueError("No successful trials completed!")

    best_result = max(successful, key=lambda r: r['score'])

    print(f"\nBest trial: {best_result['trial_id']} (seed={best_result['seed']})")
    print(f"  Score: {best_result['score']:.1f}")
    print(f"  Min separation: {best_result['min_separation_db']:.1f} dB")
    print(f"  Max separation: {best_result['max_separation_db']:.1f} dB")
    print(f"  Mean separation: {best_result['mean_separation_db']:.1f} dB")
    print(f"  Avg λ: {best_result['avg_lambda']:.3f}")
    print(f"  Median λ: {best_result['median_lambda']:.3f}")
    print(f"  Generation time: {best_result['elapsed_hours']:.2f} hours")

    # Test phase robustness on best result
    if 'patterns' in best_result:
        print("\nTesting phase robustness...")
        phase_results = test_phase_robustness(best_result['patterns'], sample_size=100)
        best_result['phase_robustness'] = phase_results

        print(f"  Ideal min: {phase_results['ideal_min_db']:.1f} dB")
        print(f"  Robust min: {phase_results['robust_min_db']:.1f} dB")
        print(f"  Degradation: {phase_results['degradation_db']:.1f} dB")

    return best_result


def save_best_patterns(best_result: Dict, output_file: str):
    """Save best pattern set to file

    Args:
        best_result: Result dict from generate_multi_trial
        output_file: Output file path
    """
    if 'patterns' not in best_result:
        raise ValueError("No patterns in result")

    save_pattern_file(best_result['patterns'], output_file)

    file_size = Path(output_file).stat().st_size
    print(f"\n✓ Saved {len(best_result['patterns'])} patterns to {output_file}")
    print(f"  File size: {file_size:,} bytes")
