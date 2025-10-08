"""Tournament-style pattern optimizer with dynamic compute allocation

Runs multiple trials in parallel with early stopping and compute redistribution
to the most promising candidates.
"""

import os
import time
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from datetime import datetime

from .trial_manager import Trial
from .elimination_strategy import EliminationStrategy, EliminationConfig

# Pattern generation imports moved to worker function to avoid
# print statements during module load that corrupt Rich UI


def simple_test_worker(x: int) -> int:
    """Simple test function to verify process pool works"""
    import os
    return x * 2 + os.getpid()


def run_single_trial_worker(trial_id: int, iterations: int, seed: int,
                           checkpoint_dir: str, p_cores: List[int] = None,
                           num_patterns: int = 16, pattern_length: int = 512,
                           target_generations: int = None, redundancy: int = 4) -> Dict[str, Any]:
    """
    Standalone function to run a single trial in a subprocess.
    This must be a module-level function to be picklable.

    Args:
        iterations: How many iterations to run THIS batch
        target_generations: Total generations this trial should reach (absolute)
        num_patterns: Number of patterns to generate (default 16)
        pattern_length: Symbols per pattern after expansion (default 512)
        redundancy: Redundancy factor - 2x, 3x, or 4x (default 4)
    """
    # Immediate debug output - use simple file operations to ensure it works
    import os
    import sys
    import traceback

    # Create debug file with absolute path and simple operations
    try:
        # Create logs parallel to checkpoints with same partitioning
        # checkpoint_dir is like "./checkpoints/p16_l512"
        # We want log_dir to be "./logs/p16_l512"
        checkpoint_parts = checkpoint_dir.split(os.sep)
        if 'checkpoints' in checkpoint_parts:
            # Replace 'checkpoints' with 'logs' in path
            log_parts = [p if p != 'checkpoints' else 'logs' for p in checkpoint_parts]
            log_dir = os.path.join(*log_parts)
        else:
            # Fallback: put logs alongside checkpoints
            log_dir = os.path.join(os.path.dirname(checkpoint_dir), 'logs')

        os.makedirs(log_dir, exist_ok=True)
        debug_file_path = os.path.join(log_dir, f'debug_trial_{trial_id}.txt')

        # Write immediately with basic file operations
        with open(debug_file_path, 'w') as f:
            f.write(f"=== WORKER START ===\n")
            f.write(f"Trial ID: {trial_id}\n")
            f.write(f"Process ID: {os.getpid()}\n")
            f.write(f"Working Directory: {os.getcwd()}\n")
            f.write(f"Python Path: {sys.path}\n")
            f.write(f"Checkpoint dir: {checkpoint_dir}\n")
            f.write(f"Iterations: {iterations}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"P-cores: {p_cores}\n\n")
            f.flush()
    except Exception as e:
        # If we can't even write debug, return error immediately
        return {
            'trial_id': trial_id,
            'iterations_run': 0,
            'best_score': 0.0,
            'final_iteration': 0,
            'convergence_rate': 0.001,
            'score_history': [],
            'error': f"Failed to create debug log: {str(e)}"
        }

    # Now continue with normal imports
    try:
        with open(debug_file_path, 'a') as f:
            f.write("Importing psutil...\n")
            f.flush()
        import psutil

        with open(debug_file_path, 'a') as f:
            f.write("Importing numpy...\n")
            f.flush()
        import numpy as np

        with open(debug_file_path, 'a') as f:
            f.write("Importing pathlib...\n")
            f.flush()
        from pathlib import Path

        with open(debug_file_path, 'a') as f:
            f.write("Importing pickle...\n")
            f.flush()
        import pickle

        with open(debug_file_path, 'a') as f:
            f.write("All standard imports successful\n\n")
            f.flush()
    except Exception as e:
        with open(debug_file_path, 'a') as f:
            f.write(f"Import error: {str(e)}\n")
            f.write(traceback.format_exc())
            f.flush()
        return {
            'trial_id': trial_id,
            'iterations_run': 0,
            'best_score': 0.0,
            'final_iteration': 0,
            'convergence_rate': 0.001,
            'score_history': [],
            'error': f"Import failed: {str(e)}"
        }

    try:
        # Add path to patterns directory for imports
        patterns_dir = Path(__file__).parent.parent.parent  # Up to patterns/
        core_dir = Path(__file__).parent  # Current core directory

        with open(debug_file_path, 'a') as f:
            f.write(f"Patterns dir: {patterns_dir}\n")
            f.write(f"Core dir: {core_dir}\n")
            f.write(f"sys.path before: {sys.path}\n")
            f.flush()

        if str(patterns_dir) not in sys.path:
            sys.path.insert(0, str(patterns_dir))
        if str(core_dir) not in sys.path:
            sys.path.insert(0, str(core_dir))

        with open(debug_file_path, 'a') as f:
            f.write(f"sys.path after: {sys.path}\n")
            f.write("Importing CASCADE modules...\n")
            f.flush()

        # No imports needed - we're discovering new patterns mathematically
        with open(debug_file_path, 'a') as f:
            f.write("Starting mathematical pattern discovery\n")
            f.flush()

        # This runs in a separate process
        # Set CPU affinity if on Windows/Linux
        try:
            p = psutil.Process()
            if p_cores:
                p.cpu_affinity(p_cores)
                p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
        except:
            pass  # Affinity setting failed, continue anyway

        # Set random seed for reproducibility
        np.random.seed(seed)

        with open(debug_file_path, 'a') as f:
            f.write(f"Set random seed to {seed}\n")
            f.flush()

        # Create checkpoint directory for this trial
        trial_checkpoint_dir = Path(checkpoint_dir) / f"trial_{trial_id}"
        trial_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with open(debug_file_path, 'a') as f:
            f.write(f"Created checkpoint dir: {trial_checkpoint_dir}\n")
            f.flush()

        # Define GA constants (before checkpoint check so always available)
        pattern_core_length = pattern_length // redundancy  # Core symbols (ternary) based on redundancy factor
        population_size = 32  # Number of pattern sets in population
        num_elites = 4  # Top performers preserved unchanged

        # Check for existing checkpoint
        checkpoint_file = trial_checkpoint_dir / "latest_checkpoint.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)

                # Check if this is a GA checkpoint or old format
                if 'population' in checkpoint:
                    # GA checkpoint
                    population = checkpoint['population']
                    fitness_scores = checkpoint.get('fitness_scores', [])
                    repetition_map = checkpoint.get('repetition_map')
                    # Deep copy pattern_set to prevent mutation issues
                    pattern_set = [p.copy() for p in checkpoint['pattern_set']]
                    start_iteration = checkpoint['iteration']
                    best_score = checkpoint['best_score']
                    score_history = checkpoint['score_history']
                    # Will continue below with GA initialization
                else:
                    # Old format - convert to GA
                    # Deep copy pattern_set to prevent mutation issues
                    pattern_set = [p.copy() for p in checkpoint['pattern_set']]
                    start_iteration = checkpoint['iteration']
                    best_score = checkpoint['best_score']
                    score_history = checkpoint['score_history']
                    # Will reinitialize population below
        else:
            with open(debug_file_path, 'a') as f:
                f.write("No checkpoint found, generating initial patterns\n")
                f.flush()

            # Initialize pattern storage
            # pattern_core_length already defined above
            pattern_set = []
            repetition_maps = []

            # Calculate Welch bound for this configuration
            min_symbols_needed = int(pattern_length * 0.625)  # 37.5% erasure tolerance
            duration_sec = pattern_length * 0.005  # At 200 symbols/sec
            welch_bound_ratio = np.sqrt((num_patterns - 1) / (pattern_length - 1))
            welch_bound_db = 20 * np.log10(welch_bound_ratio)

            with open(debug_file_path, 'a') as f:
                f.write(f"Pattern configuration:\n")
                f.write(f"  Number of patterns: {num_patterns}\n")
                f.write(f"  Pattern length: {pattern_length} symbols ({duration_sec:.2f} sec)\n")
                f.write(f"  Core length: {pattern_core_length} ternary symbols ({redundancy}x redundancy)\n")
                f.write(f"  Modulation: 3-FSK (ternary: 0,1,2 -> 3 frequencies)\n")
                f.write(f"  Min symbols for decode: {min_symbols_needed} (37.5% erasure)\n")
                f.write(f"  Welch bound: {welch_bound_db:.2f} dB (theoretical limit)\n")
                f.flush()

            # First, create repetition map (same for all patterns)
            # This defines which symbols repeat (QR-like structure)
            # Use uint16 to support core lengths > 255
            repetition_map = np.zeros(pattern_length, dtype=np.uint16)

            # Create interleaved repetition: pattern_core_length positions, each repeated Nx
            for data_pos in range(pattern_core_length):
                for rep in range(redundancy):
                    repetition_map[data_pos * redundancy + rep] = data_pos

            # Shuffle to spread repetitions (burst error resistance)
            shuffle_groups = np.arange(pattern_core_length)
            np.random.shuffle(shuffle_groups)
            shuffled_map = np.zeros(pattern_length, dtype=np.uint16)  # uint16 for large patterns
            for idx, group in enumerate(shuffle_groups):
                for rep in range(redundancy):
                    shuffled_map[idx * redundancy + rep] = group

            repetition_map = shuffled_map

            # Calculate erasure tolerance based on redundancy
            if redundancy == 4:
                erasure_tolerance = 0.375  # Need 2 of 4 copies
            elif redundancy == 3:
                erasure_tolerance = 0.33   # Need 2 of 3 copies
            else:  # redundancy == 2
                erasure_tolerance = 0.25   # Need 1 of 2 copies (50% is borderline)

            with open(debug_file_path, 'a') as f:
                f.write(f"Created repetition map: {pattern_core_length} positions × {redundancy} repetitions\n")
                f.write(f"Erasure tolerance: {erasure_tolerance:.1%}\n")
                f.flush()

        # HELPER FUNCTIONS (defined at worker level so always available regardless of checkpoint)

        # Helper function for erasure testing (uses configured erasure_tolerance)
        def test_with_erasure(p1, p2, erasure_rate=None):
            """Test correlation with random erasures"""
            if erasure_rate is None:
                erasure_rate = erasure_tolerance  # Use configured value
            keep_rate = 1.0 - erasure_rate
            mask = np.random.random(len(p1)) < keep_rate
            p1_erased = p1[mask]
            p2_erased = p2[mask]
            if len(p1_erased) < 10:
                return 0.0
            xcorr = np.correlate(p1_erased, p2_erased, mode='full')
            peak = np.max(np.abs(xcorr))
            return 20 * np.log10(peak / len(p1_erased) + 1e-10)

        # Helper function for windowed correlation testing
        def windowed_correlation(p1, p2, window_size, num_windows=8):
            """Test correlation on random windows of patterns

            Optimizes for local orthogonality instead of global orthogonality.
            This is more robust for partial pattern detection scenarios.

            Args:
                p1, p2: Full patterns (±1 valued)
                window_size: Size of window to test
                num_windows: Number of random windows to sample

            Returns:
                Worst-case correlation in dB across all windows
            """
            pattern_len = len(p1)
            worst_corr = -100.0

            for _ in range(num_windows):
                # Random window start
                if pattern_len <= window_size:
                    # Pattern smaller than window, use full pattern
                    window = slice(0, pattern_len)
                else:
                    window_start = np.random.randint(0, pattern_len - window_size)
                    window = slice(window_start, window_start + window_size)

                p1_win = p1[window]
                p2_win = p2[window]

                # Correlation on window
                xcorr = np.correlate(p1_win, p2_win, mode='full')
                peak = np.max(np.abs(xcorr))
                corr_db = 20 * np.log10(peak / len(p1_win) + 1e-10)
                worst_corr = max(worst_corr, corr_db)

            return worst_corr

        # Fitness evaluation function for a pattern set
        def evaluate_fitness(pattern_set_cores, use_sampling=True, sample_size=30, return_details=False):
            """Calculate fitness using WINDOWED orthogonality instead of global

            Optimizes for local orthogonality by testing random windows of various
            sizes. This ensures patterns are distinguishable even when only partial
            sections are received (late detection, burst interference, etc.)

            Accepts lower full-pattern correlation for better partial-pattern robustness.

            Args:
                pattern_set_cores: List of 128-bit core patterns
                use_sampling: If True, sample random pairs (faster)
                sample_size: Number of pairs to sample (default 30 of 120)
                return_details: If True, return (score, details_dict)

            Returns:
                If return_details=False: worst_case_score (float)
                If return_details=True: (worst_case_score, details_dict)
            """
            worst_normal = -100.0
            worst_flip = -100.0
            worst_erasure = -100.0

            # Window sizes to test (scaled to pattern length)
            # Test 25%, 12.5%, and 6.25% of full pattern length
            # Smaller windows weighted more heavily (prioritize short-burst detection)
            window_sizes = [
                pattern_length // 4,   # 25% window (e.g., 512 for 2048)
                pattern_length // 8,   # 12.5% window (e.g., 256 for 2048)
                pattern_length // 16   # 6.25% window (e.g., 128 for 2048)
            ]
            # Weights: prioritize smaller windows
            window_weights = {
                pattern_length // 16: 3.0,  # 128-bit (6.25%): highest weight
                pattern_length // 8: 2.0,   # 256-bit (12.5%): medium weight
                pattern_length // 4: 1.0    # 512-bit (25%): lowest weight
            }
            num_windows_per_size = 5  # Sample 5 random windows per size

            # Track per-window metrics if details requested
            # Also track weighted scores for each window
            window_scores = {ws: {'normal': -100.0, 'flip': -100.0} for ws in window_sizes}

            if return_details:
                window_metrics = {ws: {'normal': -100.0, 'flip': -100.0} for ws in window_sizes}
                global_metrics = {'normal': -100.0, 'flip': -100.0}

            # Generate all possible pairs
            all_pairs = [(i, j) for i in range(num_patterns) for j in range(i + 1, num_patterns)]

            # Select pairs to evaluate
            if use_sampling and len(all_pairs) > sample_size:
                # Random sample for speed
                pairs_to_check = [all_pairs[idx] for idx in np.random.choice(
                    len(all_pairs), sample_size, replace=False
                )]
            else:
                # Full evaluation
                pairs_to_check = all_pairs

            for i, j in pairs_to_check:
                # Expand and convert ternary {0,1,2} to {-1,0,+1} for correlation
                pi_full = pattern_set_cores[i][repetition_map]
                pj_full = pattern_set_cores[j][repetition_map]
                # 3-FSK: 0->-1, 1->0, 2->+1 (for ternary patterns)
                pi = pi_full.astype(np.float32) - 1
                pj = pj_full.astype(np.float32) - 1

                # Test multiple window sizes for local orthogonality
                for window_size in window_sizes:
                    # Normal correlation on windows
                    corr_db = windowed_correlation(pi, pj, window_size, num_windows_per_size)
                    window_scores[window_size]['normal'] = max(
                        window_scores[window_size]['normal'], corr_db
                    )
                    worst_normal = max(worst_normal, corr_db)
                    if return_details:
                        window_metrics[window_size]['normal'] = max(
                            window_metrics[window_size]['normal'], corr_db
                        )

                    # Flip correlation on windows
                    corr_f_db = windowed_correlation(pi, -pj, window_size, num_windows_per_size)
                    window_scores[window_size]['flip'] = max(
                        window_scores[window_size]['flip'], corr_f_db
                    )
                    worst_flip = max(worst_flip, corr_f_db)
                    if return_details:
                        window_metrics[window_size]['flip'] = max(
                            window_metrics[window_size]['flip'], corr_f_db
                        )

                # Keep erasure test (already tests partial patterns)
                erasure_db = test_with_erasure(pi, pj, 0.375)
                worst_erasure = max(worst_erasure, erasure_db)

                # For details: also compute global (full-pattern) correlation
                if return_details:
                    xcorr_global = np.correlate(pi, pj, mode='full')
                    peak_global = np.max(np.abs(xcorr_global))
                    global_normal = 20 * np.log10(peak_global / len(pi) + 1e-10)
                    global_metrics['normal'] = max(global_metrics['normal'], global_normal)

                    xcorr_global_f = np.correlate(pi, -pj, mode='full')
                    peak_global_f = np.max(np.abs(xcorr_global_f))
                    global_flip = 20 * np.log10(peak_global_f / len(pi) + 1e-10)
                    global_metrics['flip'] = max(global_metrics['flip'], global_flip)

            # Calculate weighted fitness (prioritize smaller windows)
            # Weighted sum: each window's worst correlation weighted by importance
            weighted_sum = 0.0
            total_weight = 0.0

            for window_size in window_sizes:
                weight = window_weights[window_size]
                worst_window = max(window_scores[window_size]['normal'],
                                  window_scores[window_size]['flip'])
                weighted_sum += weight * worst_window
                total_weight += weight

            weighted_score = weighted_sum / total_weight if total_weight > 0 else -100.0

            # Return weighted score (higher/less negative is worse)
            # Still consider erasure, but weighted score is primary
            worst_case = max(weighted_score, worst_erasure)

            if return_details:
                details = {
                    'window_metrics': window_metrics,
                    'global_metrics': global_metrics,
                    'erasure': worst_erasure,
                    'worst_normal': worst_normal,
                    'worst_flip': worst_flip,
                    'worst_overall': worst_case,
                    'weighted_score': weighted_score
                }
                return worst_case, details
            else:
                return worst_case

        # Continue with initialization or checkpoint resume
        if not checkpoint_file.exists():
            # GENETIC ALGORITHM SETUP
            # pattern_core_length, population_size, num_elites already defined above

            with open(debug_file_path, 'a') as f:
                f.write(f"Initializing genetic algorithm:\n")
                f.write(f"  Population size: {population_size} pattern sets\n")
                f.write(f"  Patterns per set: {num_patterns}\n")
                f.write(f"  Core symbols per pattern: {pattern_core_length} (ternary)\n")
                f.write(f"  Elites preserved: {num_elites}\n")
                f.flush()

            # Initialize population: 32 sets of 8 patterns (ternary cores)
            population = []
            for set_idx in range(population_size):
                pattern_set_core = []
                for i in range(num_patterns):
                    pattern_seed = seed + set_idx * 1000 + i * 7919
                    np.random.seed(pattern_seed)
                    # 3-FSK: ternary patterns {0, 1, 2} instead of binary {0, 1}
                    pattern_core = np.random.randint(0, 3, pattern_core_length, dtype=np.uint8)
                    pattern_set_core.append(pattern_core)
                population.append(pattern_set_core)

            with open(debug_file_path, 'a') as f:
                f.write(f"Generated population of {len(population)} pattern sets\n")
                f.write("Evaluating initial fitness...\n")
                f.flush()

            # Helper function for erasure testing
            def test_with_erasure(p1, p2, erasure_rate=0.375):
                """Test correlation with random erasures"""
                keep_rate = 1.0 - erasure_rate
                mask = np.random.random(len(p1)) < keep_rate
                p1_erased = p1[mask]
                p2_erased = p2[mask]
                if len(p1_erased) < 10:
                    return 0.0
                xcorr = np.correlate(p1_erased, p2_erased, mode='full')
                peak = np.max(np.abs(xcorr))
                return 20 * np.log10(peak / len(p1_erased) + 1e-10)

            # Helper function for windowed correlation testing
            def windowed_correlation(p1, p2, window_size, num_windows=8):
                """Test correlation on random windows of patterns"""
                pattern_len = len(p1)
                worst_corr = -100.0

                for _ in range(num_windows):
                    if pattern_len <= window_size:
                        window = slice(0, pattern_len)
                    else:
                        window_start = np.random.randint(0, pattern_len - window_size)
                        window = slice(window_start, window_start + window_size)

                    p1_win = p1[window]
                    p2_win = p2[window]

                    xcorr = np.correlate(p1_win, p2_win, mode='full')
                    peak = np.max(np.abs(xcorr))
                    corr_db = 20 * np.log10(peak / len(p1_win) + 1e-10)
                    worst_corr = max(worst_corr, corr_db)

                return worst_corr

            # Fitness evaluation function for a pattern set
            def evaluate_fitness(pattern_set_cores, use_sampling=True, sample_size=30, return_details=False):
                """Calculate fitness using WINDOWED orthogonality instead of global

                Args:
                    pattern_set_cores: List of 128-bit core patterns
                    use_sampling: If True, sample random pairs (faster)
                    sample_size: Number of pairs to sample (default 30 of 120)
                    return_details: If True, return (score, details_dict)
                """
                worst_normal = -100.0
                worst_flip = -100.0
                worst_erasure = -100.0

                # Window sizes for local orthogonality testing
                window_sizes = [
                    pattern_length // 4,   # 25% window
                    pattern_length // 8,   # 12.5% window
                    pattern_length // 16   # 6.25% window
                ]
                num_windows_per_size = 5

                # Track per-window metrics if details requested
                if return_details:
                    window_metrics = {ws: {'normal': -100.0, 'flip': -100.0} for ws in window_sizes}
                    global_metrics = {'normal': -100.0, 'flip': -100.0}

                # Generate all possible pairs
                all_pairs = [(i, j) for i in range(num_patterns) for j in range(i + 1, num_patterns)]

                # Select pairs to evaluate
                if use_sampling and len(all_pairs) > sample_size:
                    pairs_to_check = [all_pairs[idx] for idx in np.random.choice(
                        len(all_pairs), sample_size, replace=False
                    )]
                else:
                    pairs_to_check = all_pairs

                for i, j in pairs_to_check:
                    # Expand and convert ternary {0,1,2} to {-1,0,+1} for correlation
                    pi_full = pattern_set_cores[i][repetition_map]
                    pj_full = pattern_set_cores[j][repetition_map]
                    # 3-FSK: 0->-1, 1->0, 2->+1 (for ternary patterns)
                    pi = pi_full.astype(np.float32) - 1
                    pj = pj_full.astype(np.float32) - 1

                    # Test multiple window sizes for local orthogonality
                    for window_size in window_sizes:
                        corr_db = windowed_correlation(pi, pj, window_size, num_windows_per_size)
                        worst_normal = max(worst_normal, corr_db)
                        if return_details:
                            window_metrics[window_size]['normal'] = max(
                                window_metrics[window_size]['normal'], corr_db
                            )

                        corr_f_db = windowed_correlation(pi, -pj, window_size, num_windows_per_size)
                        worst_flip = max(worst_flip, corr_f_db)
                        if return_details:
                            window_metrics[window_size]['flip'] = max(
                                window_metrics[window_size]['flip'], corr_f_db
                            )

                    # Keep erasure test
                    erasure_db = test_with_erasure(pi, pj, 0.375)
                    worst_erasure = max(worst_erasure, erasure_db)

                    # For details: also compute global (full-pattern) correlation
                    if return_details:
                        xcorr_global = np.correlate(pi, pj, mode='full')
                        peak_global = np.max(np.abs(xcorr_global))
                        global_normal = 20 * np.log10(peak_global / len(pi) + 1e-10)
                        global_metrics['normal'] = max(global_metrics['normal'], global_normal)

                        xcorr_global_f = np.correlate(pi, -pj, mode='full')
                        peak_global_f = np.max(np.abs(xcorr_global_f))
                        global_flip = 20 * np.log10(peak_global_f / len(pi) + 1e-10)
                        global_metrics['flip'] = max(global_metrics['flip'], global_flip)

                # Return worst-case
                worst_case = max(worst_normal, worst_flip, worst_erasure)

                if return_details:
                    details = {
                        'window_metrics': window_metrics,
                        'global_metrics': global_metrics,
                        'erasure': worst_erasure,
                        'worst_normal': worst_normal,
                        'worst_flip': worst_flip,
                        'worst_overall': worst_case
                    }
                    return worst_case, details
                else:
                    return worst_case

            # Evaluate initial population (full evaluation for baseline)
            fitness_scores = []
            for set_idx, pattern_set_cores in enumerate(population):
                fitness = evaluate_fitness(pattern_set_cores, use_sampling=False)
                fitness_scores.append((fitness, set_idx))

            # Sort by fitness (lower/more negative is better)
            fitness_scores.sort(key=lambda x: x[0])
            best_score = fitness_scores[0][0]
            best_set_idx = fitness_scores[0][1]
            # IMPORTANT: Deep copy to avoid mutation when population evolves
            pattern_set = [p.copy() for p in population[best_set_idx]]
            score_history = [best_score]
            start_iteration = 0

            with open(debug_file_path, 'a') as f:
                f.write(f"Initial population fitness:\n")
                f.write(f"  Best: {fitness_scores[0][0]:.2f} dB (set {fitness_scores[0][1]})\n")
                f.write(f"  Median: {fitness_scores[16][0]:.2f} dB\n")
                f.write(f"  Worst: {fitness_scores[-1][0]:.2f} dB (set {fitness_scores[-1][1]})\n")
                f.write(f"Starting genetic algorithm optimization\n")
                f.flush()
    
        # Genetic algorithm parameters
        mutation_rate = 0.10  # 10% of symbols mutated per pattern
        crossover_rate = 0.7  # 70% of offspring use crossover

        # Calculate generations
        # Use target_generations if provided (absolute target), else calculate from batch size
        if target_generations is None:
            # Legacy: calculate from iterations parameter
            total_generations = iterations // population_size
        else:
            # New: use absolute target, run THIS batch worth of generations
            batch_generations = iterations // population_size
            total_generations = target_generations  # Absolute target for this trial

        current_iteration = start_iteration
        current_generation = start_iteration // population_size
        batch_end_generation = min(current_generation + (iterations // population_size), total_generations)

        with open(debug_file_path, 'a') as f:
            f.write(f"Genetic algorithm parameters:\n")
            f.write(f"  Total generations: {total_generations}\n")
            f.write(f"  Population size: {population_size}\n")
            f.write(f"  Mutation rate: {mutation_rate:.1%} (~{int(mutation_rate * pattern_core_length)} symbols)\n")
            f.write(f"  Crossover rate: {crossover_rate:.1%}\n")
            f.write(f"  Elites: {num_elites}\n")
            f.write(f"  Fitness metric: WINDOWED orthogonality (25%, 12.5%, 6.25% windows)\n")
            f.write(f"  Window sampling: 5 random windows per size (optimizes local orthogonality)\n")
            f.write(f"  Adaptive pair sampling: 30 pairs (75% speedup), full eval every 10th gen\n")
            f.write(f"  Expected speed: ~20 iter/s sampled, ~5 iter/s full (slower due to windowing)\n")
            f.flush()

        import time
        start_time = time.time()
        last_log_time = start_time

        # Main genetic algorithm loop (runs THIS batch of generations, then returns)
        # batch_end_generation tells us when to stop this batch
        while current_generation < batch_end_generation:
            # SELECTION: Keep top performers
            elites = [population[fitness_scores[i][1]] for i in range(num_elites)]

            # Create new population
            new_population = elites.copy()  # Preserve elites

            # CROSSOVER and MUTATION: Create offspring
            while len(new_population) < population_size:
                if np.random.random() < crossover_rate:
                    # Crossover: Breed from top 50%
                    parent1_idx = np.random.randint(0, population_size // 2)
                    parent2_idx = np.random.randint(0, population_size // 2)
                    parent1 = population[fitness_scores[parent1_idx][1]]
                    parent2 = population[fitness_scores[parent2_idx][1]]

                    # Single-point crossover: split at random pattern
                    crossover_point = np.random.randint(1, num_patterns)
                    offspring = parent1[:crossover_point] + parent2[crossover_point:]
                else:
                    # No crossover: Clone from top 50%
                    parent_idx = np.random.randint(0, population_size // 2)
                    offspring = [p.copy() for p in population[fitness_scores[parent_idx][1]]]

                # MUTATION: Mutate offspring (skip elites)
                for pattern_idx in range(num_patterns):
                    if np.random.random() < mutation_rate:
                        # Mutate this pattern
                        num_flips = np.random.randint(1, int(mutation_rate * pattern_core_length) + 5)
                        flip_positions = np.random.choice(pattern_core_length, num_flips, replace=False)
                        offspring[pattern_idx] = offspring[pattern_idx].copy()
                        # 3-FSK: Randomly change to a different ternary value {0,1,2}
                        for pos in flip_positions:
                            old_val = offspring[pattern_idx][pos]
                            # Pick a different value from {0,1,2}
                            possible_vals = [v for v in [0, 1, 2] if v != old_val]
                            offspring[pattern_idx][pos] = np.random.choice(possible_vals)

                new_population.append(offspring)

            # Replace population
            population = new_population

            # EVALUATION: Adaptive sampling (30 pairs most of the time, full every 10th)
            use_full_eval = (current_generation % 10 == 0)
            fitness_scores = []

            for set_idx, pattern_set_cores in enumerate(population):
                if use_full_eval:
                    # Full evaluation every 10th generation for accurate tracking
                    fitness = evaluate_fitness(pattern_set_cores, use_sampling=False)
                else:
                    # Sampled evaluation (30 pairs) for speed
                    fitness = evaluate_fitness(pattern_set_cores, use_sampling=True, sample_size=30)
                fitness_scores.append((fitness, set_idx))

            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[0])

            # Track best (only update on full evaluations to avoid sampling noise)
            current_best = fitness_scores[0][0]
            if use_full_eval:
                # Full evaluation - safe to update historically best pattern and score
                if current_best < best_score:
                    best_score = current_best
                    best_set_idx = fitness_scores[0][1]
                    # IMPORTANT: Deep copy to avoid mutation when population evolves
                    pattern_set = [p.copy() for p in population[best_set_idx]]
            # On sampled evals, keep historical best_score and pattern_set unchanged

            score_history.append(best_score)

            # Update iteration counter
            current_iteration = current_generation * population_size
            current_generation += 1

            # Logging and checkpointing
            current_time = time.time()
            if current_generation % 100 == 0 or (current_time - last_log_time) >= 10:
                elapsed = current_time - start_time
                if elapsed > 0:
                    gen_per_sec = (current_generation - start_iteration // population_size) / elapsed
                    iter_per_sec = current_iteration / elapsed if current_iteration > 0 else 0
                    speed_str = f"{gen_per_sec:.2f} gen/s ({iter_per_sec:.1f} iter/s)"
                else:
                    speed_str = "N/A"

                # Population diversity
                best_fitness = fitness_scores[0][0]
                median_fitness = fitness_scores[population_size // 2][0]
                worst_fitness = fitness_scores[-1][0]

                eval_type = "FULL" if (current_generation % 10 == 0) else "sampled(30)"

                # Get detailed metrics for historically best pattern (every 100 generations)
                detailed_metrics = None
                if current_generation % 100 == 0:
                    # Use pattern_set (historically best) for consistent tracking
                    _, detailed_metrics = evaluate_fitness(pattern_set, use_sampling=False, return_details=True)

                with open(debug_file_path, 'a') as f:
                    f.write(f"Generation {current_generation}/{total_generations} [{eval_type}]:\n")
                    f.write(f"  Best: {best_fitness:.2f} dB, Median: {median_fitness:.2f} dB, "
                           f"Worst: {worst_fitness:.2f} dB\n")
                    f.write(f"  Overall best: {best_score:.2f} dB\n")

                    # Show detailed window breakdown every 100 generations
                    if detailed_metrics:
                        f.write(f"  Window Orthogonality Breakdown:\n")
                        window_sizes = sorted(detailed_metrics['window_metrics'].keys(), reverse=True)
                        for ws in window_sizes:
                            wm = detailed_metrics['window_metrics'][ws]
                            percent = (ws / pattern_length) * 100
                            worst_win = max(wm['normal'], wm['flip'])
                            f.write(f"    {ws:4d}bit ({percent:4.1f}%): {worst_win:6.2f} dB "
                                   f"(normal={wm['normal']:6.2f}, flip={wm['flip']:6.2f})\n")
                        gm = detailed_metrics['global_metrics']
                        worst_global = max(gm['normal'], gm['flip'])
                        f.write(f"    GLOBAL ({pattern_length}bit): {worst_global:6.2f} dB "
                               f"(normal={gm['normal']:6.2f}, flip={gm['flip']:6.2f})\n")
                        f.write(f"    Erasure test: {detailed_metrics['erasure']:6.2f} dB\n")

                    f.write(f"  Elapsed: {elapsed:.1f}s, Speed: {speed_str}\n")
                    f.flush()
                last_log_time = current_time

            # Save checkpoint every 100 iterations for live updates (skip iteration 0)
            if (current_iteration > 0 and current_iteration % 100 == 0) or current_generation >= total_generations:
                # Get detailed metrics for checkpoint (use historically best pattern_set)
                try:
                    # Use pattern_set which is the historically best (never gets worse)
                    _, detailed_metrics = evaluate_fitness(pattern_set, use_sampling=False, return_details=True)

                    checkpoint = {
                        'pattern_set': pattern_set,  # Best pattern set (128-bit cores)
                        'repetition_map': repetition_map,  # Single shared map
                        'population': population,  # Full population for resume
                        'fitness_scores': fitness_scores,  # Current fitness ranking
                        'iteration': current_iteration,
                        'generation': current_generation,
                        'best_score': best_score,
                        'score_history': score_history,
                        'trial_id': trial_id,
                        'seed': seed,
                        'window_metrics': detailed_metrics['window_metrics'],
                        'global_metrics': detailed_metrics['global_metrics'],
                        'erasure_metrics': detailed_metrics['erasure'],
                        'weighted_score': detailed_metrics.get('weighted_score', best_score)
                    }
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint, f)
                    # Debug: confirm metrics saved
                    with open(debug_file_path, 'a') as f:
                        f.write(f"  Saved checkpoint with window metrics: {list(detailed_metrics['window_metrics'].keys())}\n")
                        f.flush()
                except Exception as e:
                    # Log error but don't crash - save checkpoint without metrics
                    with open(debug_file_path, 'a') as f:
                        f.write(f"Warning: Failed to compute detailed metrics for checkpoint: {e}\n")
                        f.flush()
                    # Save checkpoint without detailed metrics
                    checkpoint = {
                        'pattern_set': pattern_set,
                        'repetition_map': repetition_map,
                        'population': population,
                        'fitness_scores': fitness_scores,
                        'iteration': current_iteration,
                        'generation': current_generation,
                        'best_score': best_score,
                        'score_history': score_history,
                        'trial_id': trial_id,
                        'seed': seed
                    }
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint, f)
    
        # Calculate convergence rate based on score improvement
        if len(score_history) > 10:
            recent_scores = score_history[-10:]
            older_scores = score_history[-20:-10] if len(score_history) >= 20 else score_history[:10]
    
            # Calculate average improvement
            if all(np.isfinite(recent_scores)) and all(np.isfinite(older_scores)):
                recent_avg = np.mean(recent_scores)
                older_avg = np.mean(older_scores)
                score_diff = older_avg - recent_avg  # Improvement (positive is good)
                convergence_rate = max(0.001, abs(score_diff) / 10)
            else:
                convergence_rate = 0.001
        else:
            convergence_rate = 0.001
    
        # NESTED PATTERN EXTRACTION
        # Generate multiple length variants from the optimized core patterns
        # Shorter patterns are prefixes of longer ones (perfect cross-length orthogonality)

        with open(debug_file_path, 'a') as f:
            f.write(f"\n=== EXTRACTING NESTED PATTERNS ===\n")
            f.write(f"Full core length: {pattern_core_length} bits\n")
            f.flush()

        # Determine nested lengths (powers of 2, down to 128 minimum)
        nested_core_lengths = []
        current_len = pattern_core_length
        while current_len >= 64:  # Minimum 64 core bits
            nested_core_lengths.append(current_len)
            current_len = current_len // 2

        nested_patterns = {}
        nested_orthogonality = {}

        for core_len in nested_core_lengths:
            # Extract prefix of each pattern
            variant_cores = [p[:core_len] for p in pattern_set]

            # Calculate expanded length for this variant
            variant_full_len = core_len * redundancy

            # Create repetition map for this length
            variant_rep_map = np.zeros(variant_full_len, dtype=np.uint16)
            for data_pos in range(core_len):
                for rep in range(redundancy):
                    variant_rep_map[data_pos * redundancy + rep] = data_pos

            # Shuffle (use same seed for consistency)
            np.random.seed(seed)
            shuffle_groups = np.arange(core_len)
            np.random.shuffle(shuffle_groups)
            shuffled_variant_map = np.zeros(variant_full_len, dtype=np.uint16)
            for idx, group in enumerate(shuffle_groups):
                for rep in range(redundancy):
                    shuffled_variant_map[idx * redundancy + rep] = group

            # Test orthogonality at this length
            worst_corr = -100.0
            for i in range(num_patterns):
                for j in range(i + 1, num_patterns):
                    pi_full = variant_cores[i][shuffled_variant_map]
                    pj_full = variant_cores[j][shuffled_variant_map]
                    pi = 2 * pi_full.astype(np.float32) - 1
                    pj = 2 * pj_full.astype(np.float32) - 1

                    # Normal + flip correlation
                    xcorr_n = np.correlate(pi, pj, mode='full')
                    xcorr_f = np.correlate(pi, -pj, mode='full')
                    peak = max(np.max(np.abs(xcorr_n)), np.max(np.abs(xcorr_f)))
                    corr_db = 20 * np.log10(peak / variant_full_len + 1e-10)
                    worst_corr = max(worst_corr, corr_db)

            nested_patterns[variant_full_len] = {
                'cores': variant_cores,
                'repetition_map': shuffled_variant_map,
                'core_length': core_len,
                'full_length': variant_full_len
            }
            nested_orthogonality[variant_full_len] = worst_corr

            with open(debug_file_path, 'a') as f:
                f.write(f"  Length {variant_full_len}: {core_len} core bits, "
                       f"orthogonality {worst_corr:.2f} dB\n")
                f.flush()

        # Save final pattern set with ALL nested variants
        final_patterns_file = trial_checkpoint_dir / f"final_patterns_{current_iteration}.pkl"

        final_data = {
            'nested_patterns': nested_patterns,  # Dict of {length: pattern_data}
            'nested_orthogonality': nested_orthogonality,  # {length: dB_score}
            'num_patterns': len(pattern_set),
            'max_core_length': pattern_core_length,
            'max_full_length': pattern_length,
            'redundancy': redundancy,
            'best_score': best_score,
            'trial_id': trial_id,
            'seed': seed,
            'iterations': current_iteration,
            'algorithm': 'genetic_nested'
        }
        with open(final_patterns_file, 'wb') as f:
            pickle.dump(final_data, f)

        total_elapsed = time.time() - start_time
        iterations_done = current_iteration - start_iteration
        avg_speed = iterations_done / total_elapsed if total_elapsed > 0 else float('inf')

        # Calculate final detailed metrics for return to main process
        # Use pattern_set which is the historically best (never gets worse)
        try:
            _, final_detailed_metrics = evaluate_fitness(pattern_set, use_sampling=False, return_details=True)
        except:
            final_detailed_metrics = None

        with open(debug_file_path, 'a') as f:
            f.write(f"\n=== WORKER COMPLETE ===\n")
            f.write(f"Final iteration: {current_iteration}\n")
            f.write(f"Iterations completed: {iterations_done}\n")
            f.write(f"Final best score: {best_score:.2f} dB\n")
            f.write(f"Convergence rate: {convergence_rate:.6f}\n")
            f.write(f"Total time: {total_elapsed:.1f} seconds\n")
            if total_elapsed > 0:
                f.write(f"Average speed: {avg_speed:.1f} iter/s\n")
            else:
                f.write(f"Average speed: N/A (completed instantly)\n")
            f.write(f"Patterns saved to: {final_patterns_file}\n")
            f.flush()

        # Return results with final metrics
        result = {
            'trial_id': trial_id,
            'iterations_run': current_iteration - start_iteration,
            'best_score': best_score,
            'final_iteration': current_iteration,
            'convergence_rate': convergence_rate,
            'score_history': score_history[-100:],  # Last 100 scores
            'patterns_file': str(final_patterns_file)
        }

        # Include metrics if available
        if final_detailed_metrics:
            result['window_metrics'] = final_detailed_metrics['window_metrics']
            result['global_metrics'] = final_detailed_metrics['global_metrics']
            result['erasure_metrics'] = final_detailed_metrics['erasure']
            result['weighted_score'] = final_detailed_metrics.get('weighted_score', best_score)

        return result

    except Exception as e:
        # Log error to file for debugging
        error_file = Path(checkpoint_dir).parent / 'logs' / f'error_trial_{trial_id}.txt'
        error_file.parent.mkdir(exist_ok=True, parents=True)
        with open(error_file, 'w') as f:
            import traceback
            f.write(f"Error in trial {trial_id}:\n")
            f.write(str(e) + "\n\n")
            f.write(traceback.format_exc())

        # Return error result so tournament doesn't hang
        return {
            'trial_id': trial_id,
            'iterations_run': 0,
            'best_score': 0.0,
            'final_iteration': 0,
            'convergence_rate': 0.001,
            'score_history': [],
            'error': str(e)
        }


class DynamicTournamentOptimizer:
    """Tournament optimizer with dynamic compute allocation"""

    def __init__(
        self,
        total_generations: int = 150_000,  # Total generations across all trials
        num_initial_trials: int = 8,
        checkpoint_dir: str = "./checkpoints",
        log_callback: Optional[Callable] = None,
        execution_mode: str = "auto",  # "process", "thread", "sequential", or "auto"
        num_patterns: int = 16,  # Number of patterns to generate
        pattern_length: int = 512,  # Symbols per pattern (after expansion)
        redundancy: int = 4  # Redundancy factor (2, 3, or 4)
    ):
        self.total_generations = total_generations
        self.num_initial_trials = num_initial_trials
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_callback = log_callback or print
        self.execution_mode = execution_mode
        self.num_patterns = num_patterns
        self.pattern_length = pattern_length
        self.redundancy = redundancy

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.elimination_strategy = EliminationStrategy()
        self.trials: List[Trial] = []
        self.active_trials: List[int] = []

        # Tracking
        self.compute_used = 0  # Track iterations (for compatibility)
        self.current_generation = 0  # Track total generations across trials
        self.global_best_score = float('inf')
        self.global_best_trial_id = None
        self.start_time = None
        self.current_phase = 'exploration'
        self.running = False
        self.monitor_thread = None

    def initialize_trials(self):
        """Initialize all trials"""
        self.trials = []
        # Calculate target generations per trial
        target_gens_per_trial = self.total_generations // self.num_initial_trials

        for i in range(self.num_initial_trials):
            trial = Trial(
                trial_id=i,
                seed=1000 * (i + 1),
                p_cores=self._assign_p_cores(i)
            )
            # Set target generations for this trial
            trial.target_generations = target_gens_per_trial
            trial.target_iterations = target_gens_per_trial * 32  # For compatibility
            self.trials.append(trial)

        self.active_trials = list(range(self.num_initial_trials))
        self.log_callback(f"Initialized {self.num_initial_trials} trials, "
                         f"{target_gens_per_trial:,} generations each")

    def _assign_p_cores(self, trial_id: int) -> List[int]:
        """Assign P-cores to a trial"""
        # For 8 trials on 8 P-cores, each gets 1 core initially
        if self.num_initial_trials <= 8:
            return [trial_id]

        # For more trials, share cores
        cores_per_trial = 8 // min(self.num_initial_trials, 8)
        start_core = (trial_id * cores_per_trial) % 8
        return list(range(start_core, min(start_core + cores_per_trial, 8)))

    def _start_checkpoint_monitor(self):
        """Start background thread to monitor checkpoints for live updates"""
        import threading
        import time

        def monitor_checkpoints():
            while self.running:
                try:
                    for trial_id in self.active_trials[:]:  # Copy to avoid modification during iteration
                        self._update_trial_from_checkpoint(trial_id)
                except Exception:
                    pass  # Ignore errors in background thread
                time.sleep(2)  # Poll every 2 seconds

        self.running = True
        self.monitor_thread = threading.Thread(target=monitor_checkpoints, daemon=True)
        self.monitor_thread.start()

    def run_tournament(self) -> List[np.ndarray]:
        """Run the tournament optimization"""
        self.start_time = datetime.now()
        self.initialize_trials()

        self.log_callback("=" * 60)
        self.log_callback("Starting CASCADE Pattern Tournament")
        self.log_callback(f"Total generations: {self.total_generations:,}")
        self.log_callback(f"Trials: {self.num_initial_trials} × {self.total_generations // self.num_initial_trials:,} generations each")
        self.log_callback("=" * 60)

        # Start background checkpoint monitoring for live updates
        self._start_checkpoint_monitor()

        # Main tournament loop - run trials until complete
        # For GA: trials run until they hit their generation target (no budget checks)
        loop_count = 0
        last_compute_used = 0
        stuck_count = 0

        while len(self.active_trials) > 0:
            loop_count += 1

            # Safety: Detect infinite loops (no progress)
            if self.compute_used == last_compute_used:
                stuck_count += 1
                if stuck_count > 3:
                    self.log_callback("ERROR: Loop stuck - no progress for 3 iterations")
                    self.log_callback(f"Active trials: {self.active_trials}")
                    for tid in self.active_trials:
                        t = self.trials[tid]
                        self.log_callback(f"  Trial {tid}: gen {t.iterations//32}/{t.target_generations}, status={t.status}")
                    break
            else:
                stuck_count = 0
                last_compute_used = self.compute_used

            if loop_count % 10 == 0:  # Only log every 10th iteration to reduce spam
                self.log_callback(f"DEBUG: Main loop iteration {loop_count}, active_trials: {self.active_trials}")

            # Update phase
            self._update_phase()

            # Run active trials
            self._run_trial_batch()

            if loop_count % 10 == 0:
                self.log_callback(f"DEBUG: After _run_trial_batch, active_trials: {self.active_trials}")

            # Check if any trials exhausted (completed all generations)
            completed_trials = []
            for trial_id in self.active_trials[:]:
                trial = self.trials[trial_id]
                # Calculate generation progress
                total_gens = trial.target_generations
                current_gen = trial.iterations // 32

                # Debug log
                self.log_callback(f"DEBUG: Trial {trial_id}: gen {current_gen}/{total_gens}, "
                                 f"iterations {trial.iterations}/{trial.target_iterations}, "
                                 f"status={trial.status}")

                if trial.status == 'completed':
                    completed_trials.append(trial_id)

            # Remove completed trials from active list
            for trial_id in completed_trials:
                self.active_trials.remove(trial_id)
                self.log_callback(f"Trial {trial_id} completed all generations")

            # Debug: Show active trials
            self.log_callback(f"DEBUG: Active trials remaining: {self.active_trials}")

        # Stop monitoring
        self.running = False

        # Finalize and return best patterns
        return self._finalize_tournament()

    def _update_phase(self):
        """Update optimization phase based on generation progress"""
        # Use best trial's generation progress
        if self.trials:
            best_trial = min(self.trials, key=lambda t: t.best_score)
            if hasattr(best_trial, 'target_generations'):
                progress = (best_trial.iterations // 32) / best_trial.target_generations
            else:
                progress = 0
        else:
            progress = 0

        if progress < 0.25:
            new_phase = 'exploration'
        elif progress < 0.5:
            new_phase = 'evaluation'
        elif progress < 0.75:
            new_phase = 'exploitation'
        else:
            new_phase = 'refinement'

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.elimination_strategy.adjust_aggressiveness(new_phase)
            self.log_callback(f"\n>>> Entering {new_phase.upper()} phase\n")

    def _update_trial_from_checkpoint(self, trial_id: int):
        """Update trial statistics from its checkpoint file (for live updates)"""
        trial = self.trials[trial_id]
        checkpoint_file = self.checkpoint_dir / f"trial_{trial_id}" / "latest_checkpoint.pkl"

        if checkpoint_file.exists():
            try:
                import pickle
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)

                # Update trial stats if checkpoint has newer data
                if checkpoint.get('iteration', 0) > trial.iterations:
                    trial.iterations = checkpoint['iteration']
                    trial.best_score = float(checkpoint.get('best_score', trial.best_score))
                    trial.current_score = trial.best_score

                    # Window metrics (for display)
                    if 'window_metrics' in checkpoint:
                        trial.window_metrics = checkpoint['window_metrics']
                        # Debug: verify metrics were loaded
                        if trial.window_metrics and len(trial.window_metrics) > 0:
                            print(f"✓ Loaded window metrics for trial {trial_id}: {list(trial.window_metrics.keys())}")
                    if 'global_metrics' in checkpoint:
                        trial.global_metrics = checkpoint['global_metrics']
                    if 'erasure_metrics' in checkpoint:
                        trial.erasure_metrics = checkpoint['erasure_metrics']
                    if 'weighted_score' in checkpoint:
                        trial.weighted_score = checkpoint['weighted_score']

                    # GA-specific stats (for display)
                    if 'generation' in checkpoint:
                        trial.generation = checkpoint['generation']
                        trial.algorithm = 'GA'
                    if 'fitness_scores' in checkpoint:
                        # Extract population diversity
                        fitness_scores = checkpoint['fitness_scores']
                        if len(fitness_scores) >= 3:
                            trial.ga_best = fitness_scores[0][0]
                            trial.ga_median = fitness_scores[len(fitness_scores)//2][0]
                            trial.ga_worst = fitness_scores[-1][0]

                    # Update progress
                    total_gens = trial.target_generations if hasattr(trial, 'target_generations') else (trial.target_iterations // 32)
                    current_gen = trial.iterations // 32
                    trial.progress = current_gen / total_gens if total_gens > 0 else 0
                    remaining_gens = max(0, total_gens - current_gen)
                    trial.calculate_eta(remaining_gens * 32)

                    # Update convergence rate
                    score_history = checkpoint.get('score_history', [])
                    if len(score_history) > 10:
                        recent = score_history[-10:]
                        older = score_history[-20:-10] if len(score_history) >= 20 else score_history[:10]
                        if all(np.isfinite(recent)) and all(np.isfinite(older)):
                            import numpy as np
                            recent_avg = np.mean(recent)
                            older_avg = np.mean(older)
                            score_diff = older_avg - recent_avg
                            trial.convergence_rate = max(0.001, abs(score_diff) / 10)

                    # Update global best
                    if trial.best_score < self.global_best_score:
                        self.global_best_score = trial.best_score
                        self.global_best_trial_id = trial_id
                        trial.is_best = True
                        for t in self.trials:
                            if t.trial_id != trial_id:
                                t.is_best = False

                    # Update compute used based on actual trial iterations
                    # This ensures compute_used reflects real-time progress
                    self.compute_used = sum(t.iterations for t in self.trials)
            except Exception:
                pass  # Ignore checkpoint read errors

    def _run_trial_batch(self):
        """Run all active trials for eval_interval iterations"""
        if not self.active_trials:
            self.log_callback("DEBUG: _run_trial_batch called but no active trials!")
            return

        self.log_callback(f"DEBUG: _run_trial_batch starting with {len(self.active_trials)} active trials")

        # Prepare batch execution
        batch_size = min(len(self.active_trials), 8)  # Max 8 parallel workers

        # Determine execution strategy
        if self.execution_mode == "sequential":
            executor_class = None
        elif self.execution_mode == "thread":
            executor_class = ThreadPoolExecutor
        elif self.execution_mode == "process":
            executor_class = ProcessPoolExecutor
        else:  # auto mode
            # Try process first, fall back to thread, then sequential
            try:
                # Test if ProcessPoolExecutor works with our simple test function
                with ProcessPoolExecutor(max_workers=1) as test_executor:
                    test_future = test_executor.submit(simple_test_worker, 21)
                    test_result = test_future.result(timeout=2)
                    if test_result > 42:  # Should be 42 + pid
                        executor_class = ProcessPoolExecutor
                        self.log_callback("Using ProcessPoolExecutor for parallel trials")
                    else:
                        raise Exception("Test worker failed")
            except Exception as e:
                try:
                    # Fall back to ThreadPoolExecutor
                    executor_class = ThreadPoolExecutor
                    self.log_callback(f"Process pool failed ({type(e).__name__}: {str(e)}), using ThreadPoolExecutor")
                except:
                    executor_class = None
                    self.log_callback("Both process and thread pools failed, using sequential execution")

        # Run trials with selected executor
        if executor_class:
            try:
                # Create master debug log to track submissions
                master_debug = self.checkpoint_dir.parent / 'logs' / 'master_debug.txt'
                master_debug.parent.mkdir(exist_ok=True, parents=True)

                with open(master_debug, 'a') as f:
                    f.write(f"\n=== BATCH RUN START ===\n")
                    f.write(f"Time: {datetime.now()}\n")
                    f.write(f"Executor: {executor_class.__name__}\n")
                    f.write(f"Batch size: {batch_size}\n")
                    f.write(f"Active trials: {self.active_trials}\n")
                    f.flush()

                with executor_class(max_workers=batch_size) as executor:
                    futures = {}

                    for trial_id in self.active_trials:
                        trial = self.trials[trial_id]

                        # Skip if already completed
                        if trial.status == 'completed':
                            continue

                        # For GA: Run in generation batches
                        # First run: 200k iterations (6,250 generations)
                        # Subsequent: 50k iterations (1,562 generations) until target reached
                        total_generations_for_trial = trial.target_generations
                        current_gen_for_trial = trial.iterations // 32

                        if current_gen_for_trial >= total_generations_for_trial:
                            # This trial finished all its generations
                            trial.status = 'completed'
                            with open(master_debug, 'a') as f:
                                f.write(f"Trial {trial_id} completed {current_gen_for_trial} generations\n")
                                f.flush()
                            continue

                        # Submit trial to run to completion (all remaining generations)
                        # Workers save checkpoints every 1000 iterations for live UI updates
                        iterations_to_run = trial.target_iterations - trial.iterations

                        with open(master_debug, 'a') as f:
                            f.write(f"Trial {trial_id}: Gen {current_gen_for_trial}/{total_generations_for_trial}, "
                                   f"running {iterations_to_run} iterations\n")
                            f.flush()

                        # Update trial status to running
                        trial.status = 'running'
                        trial.start()

                        with open(master_debug, 'a') as f:
                            f.write(f"Submitting trial {trial_id}:\n")
                            f.write(f"  Iterations to run: {iterations_to_run}\n")
                            f.write(f"  Seed: {trial.seed}\n")
                            f.write(f"  P-cores: {trial.p_cores}\n")
                            f.flush()

                        future = executor.submit(
                            run_single_trial_worker,
                            trial_id,
                            iterations_to_run,
                            trial.seed,
                            str(self.checkpoint_dir),
                            trial.p_cores,
                            self.num_patterns,
                            self.pattern_length,
                            trial.target_generations,
                            self.redundancy
                        )
                        futures[future] = trial_id

                        with open(master_debug, 'a') as f:
                            f.write(f"  Future created for trial {trial_id}\n")
                            f.flush()

                    # Collect results
                    with open(master_debug, 'a') as f:
                        f.write(f"Waiting for {len(futures)} futures to complete...\n")
                        f.flush()

                    for future in as_completed(futures):
                        trial_id = futures[future]
                        with open(master_debug, 'a') as f:
                            f.write(f"Future completed for trial {trial_id}\n")
                            f.flush()
                        try:
                            result = future.result(timeout=300)  # 5 minute timeout
                            with open(master_debug, 'a') as f:
                                f.write(f"Got result for trial {trial_id}: {result.get('iterations_run', 0)} iterations\n")
                                f.flush()
                            self._process_trial_result(trial_id, result)
                        except Exception as e:
                            with open(master_debug, 'a') as f:
                                f.write(f"ERROR in trial {trial_id}: {e}\n")
                                import traceback
                                f.write(traceback.format_exc())
                                f.flush()
                            self.log_callback(f"Error in trial {trial_id}: {e}")

            except Exception as e:
                self.log_callback(f"Executor failed during execution: {e}")
                executor_class = None  # Fall back to sequential

        # Fallback or explicit sequential execution
        if not executor_class:
            if self.execution_mode != "sequential":
                self.log_callback("WARNING: Running trials sequentially (slower)")

            for trial_id in self.active_trials:
                trial = self.trials[trial_id]

                # Skip if completed
                if trial.status == 'completed':
                    continue

                # Check if trial finished its generations
                total_generations_for_trial = trial.target_generations
                current_gen_for_trial = trial.iterations // 32

                if current_gen_for_trial >= total_generations_for_trial:
                    trial.status = 'completed'
                    continue

                # Update trial status to running
                trial.status = 'running'
                trial.start()

                try:
                    # Run the trial to completion (all remaining generations)
                    iterations_to_run = trial.target_iterations - trial.iterations

                    result = run_single_trial_worker(
                        trial_id,
                        iterations_to_run,
                        trial.seed,
                        str(self.checkpoint_dir),
                        trial.p_cores,
                        self.num_patterns,
                        self.pattern_length,
                        trial.target_generations,
                        self.redundancy
                    )
                    self._process_trial_result(trial_id, result)
                except Exception as e:
                    self.log_callback(f"Error in trial {trial_id}: {e}")


    def _process_trial_result(self, trial_id: int, result: Dict[str, Any]):
        """Process results from a trial run"""
        trial = self.trials[trial_id]

        # Check for errors
        if 'error' in result:
            trial.status = 'error'
            self.log_callback(f"ERROR in trial {trial_id}: {result['error']}")
            self.log_callback(f"Check logs/error_trial_{trial_id}.txt for details")
            return

        # Update trial state
        trial.iterations = result['final_iteration']
        trial.best_score = float(result['best_score'])  # Ensure it's a float
        trial.current_score = trial.best_score
        trial.convergence_rate = result['convergence_rate']
        trial.score_history.extend(result['score_history'])

        # Store patterns file path if available
        if 'patterns_file' in result:
            trial.patterns_file = result['patterns_file']

        # Store metrics for UI display
        if 'window_metrics' in result:
            trial.window_metrics = result['window_metrics']
        if 'global_metrics' in result:
            trial.global_metrics = result['global_metrics']
        if 'erasure_metrics' in result:
            trial.erasure_metrics = result['erasure_metrics']
        if 'weighted_score' in result:
            trial.weighted_score = result['weighted_score']

        # Update trial status based on generation completion
        total_gens = trial.target_generations
        current_gen = trial.iterations // 32

        if current_gen >= total_gens:
            trial.status = 'completed'
        else:
            trial.status = 'paused'  # Waiting for next batch

        # Update global best
        if trial.best_score < self.global_best_score:
            self.global_best_score = trial.best_score
            self.global_best_trial_id = trial_id
            trial.is_best = True

            # Mark others as not best
            for t in self.trials:
                if t.trial_id != trial_id:
                    t.is_best = False

        # Update compute used based on actual trial iterations
        self.compute_used = sum(t.iterations for t in self.trials)

        # Update progress based on generations
        trial.progress = current_gen / total_gens if total_gens > 0 else 1.0
        remaining_gens = total_gens - current_gen
        trial.calculate_eta(remaining_gens * 32)  # Convert generations to iterations for ETA

        # Log update
        self.log_callback(
            f"Trial {trial_id}: Iteration {trial.iterations:,} | "
            f"Best: {trial.best_score:.2f} dB | "
            f"Conv: {trial.convergence_rate:.4f}"
        )

    def _evaluate_and_eliminate(self):
        """Evaluate trials and eliminate underperformers"""
        # Get active trials
        active_trial_objects = [t for t in self.trials if t.trial_id in self.active_trials]

        # Identify underperformers
        to_eliminate = self.elimination_strategy.identify_underperformers(
            active_trial_objects,
            self.compute_used
        )

        if not to_eliminate:
            return

        # Process eliminations
        eliminated_ids = []
        for trial_id, reason in to_eliminate:
            trial = self.trials[trial_id]
            trial.eliminate(reason)
            self.active_trials.remove(trial_id)
            eliminated_ids.append(trial_id)

            self.log_callback(f"\n⛔ Eliminating Trial {trial_id}")
            self.log_callback(f"   Reason: {reason}")
            self.log_callback(f"   Final score: {trial.best_score:.2f} dB\n")

        # Reallocate compute to survivors
        if self.active_trials:
            surviving_trials = [t for t in self.trials if t.trial_id in self.active_trials]
            remaining_budget = self.total_budget - self.compute_used

            allocation = self.elimination_strategy.compute_reallocation(
                eliminated_ids,
                surviving_trials,
                remaining_budget
            )

            # Bonus allocation disabled for GA (all trials run to target_generations)
            # for trial_id, bonus in allocation.items():
            #     self.trials[trial_id].bonus_budget += bonus
            #     self.log_callback(f"Trial {trial_id} receives {bonus:,} bonus iterations")

    def _check_convergence(self) -> bool:
        """Check if tournament has converged - DISABLED, always run to completion"""
        # Never stop early - always run full budget
        return False

    def _finalize_tournament(self) -> List[np.ndarray]:
        """Finalize tournament and return best patterns"""
        self.log_callback("\n" + "=" * 60)
        self.log_callback("Tournament Complete!")
        self.log_callback("=" * 60)

        # Find best trial
        best_trial = min(self.trials, key=lambda t: t.best_score)

        self.log_callback(f"\n🏆 Winner: Trial {best_trial.trial_id}")
        self.log_callback(f"   Final score: {best_trial.best_score:.2f} dB")
        self.log_callback(f"   Total iterations: {best_trial.iterations:,}")
        self.log_callback(f"   Seed: {best_trial.seed}")

        # Generate report
        self._generate_final_report()

        # Load and return best patterns with repetition maps
        if hasattr(best_trial, 'patterns_file') and best_trial.patterns_file:
            import pickle
            try:
                with open(best_trial.patterns_file, 'rb') as f:
                    pattern_data = pickle.load(f)

                    # Check for nested pattern format
                    if isinstance(pattern_data, dict) and 'nested_patterns' in pattern_data:
                        # New nested format
                        nested_patterns = pattern_data['nested_patterns']
                        nested_orth = pattern_data['nested_orthogonality']

                        self.log_callback(f"\n✅ Loaded nested pattern set with {pattern_data['num_patterns']} patterns")
                        self.log_callback(f"Redundancy: {pattern_data['redundancy']}x")
                        self.log_callback(f"\nAvailable pattern lengths:")

                        for length in sorted(nested_patterns.keys(), reverse=True):
                            orth = nested_orth[length]
                            core_bits = nested_patterns[length]['core_length']
                            duration = length * 0.005  # @ 200 sym/s
                            self.log_callback(f"  {length:4d} symbols ({core_bits:3d} core): {orth:6.2f} dB ({duration:.2f}s)")

                        return pattern_data
                    elif isinstance(pattern_data, dict):
                        # Old non-nested format
                        patterns = pattern_data.get('patterns', [])
                        self.log_callback(f"\nLoaded {len(patterns)} patterns (old format)")
                        return pattern_data
                    else:
                        # Very old format - just patterns
                        self.log_callback(f"\nLoaded {len(pattern_data)} patterns (legacy format)")
                        return {'patterns': pattern_data, 'repetition_maps': []}
            except Exception as e:
                self.log_callback(f"Error loading patterns: {e}")
                return {'patterns': [], 'repetition_maps': []}
        else:
            # Try to load from checkpoint directory
            checkpoint_dir = self.checkpoint_dir / f"trial_{best_trial.trial_id}"
            pattern_files = sorted(checkpoint_dir.glob("final_patterns_*.pkl"))
            if pattern_files:
                import pickle
                with open(pattern_files[-1], 'rb') as f:
                    pattern_data = pickle.load(f)
                    if isinstance(pattern_data, dict):
                        patterns = pattern_data.get('patterns', [])
                        self.log_callback(f"\nLoaded {len(patterns)} patterns with repetition maps from checkpoint")
                        return pattern_data
                    else:
                        self.log_callback(f"\nLoaded {len(pattern_data)} patterns from checkpoint (old format)")
                        return {'patterns': pattern_data, 'repetition_maps': []}
            else:
                self.log_callback("No patterns file found")
                return {'patterns': [], 'repetition_maps': []}

    def _generate_final_report(self):
        """Generate comprehensive final report"""
        report_path = self.checkpoint_dir / "tournament_report.txt"

        with open(report_path, "w") as f:
            f.write("CASCADE Pattern Tournament Report\n")
            f.write("=" * 60 + "\n\n")

            # Overall statistics
            runtime = (datetime.now() - self.start_time).total_seconds() / 3600
            f.write(f"Runtime: {runtime:.2f} hours\n")
            f.write(f"Total compute used: {self.compute_used:,} iterations\n")
            f.write(f"Best score achieved: {self.global_best_score:.2f} dB\n")
            f.write(f"Winner: Trial {self.global_best_trial_id}\n\n")

            # Trial summary
            f.write("Trial Summary:\n")
            f.write("-" * 60 + "\n")
            for trial in sorted(self.trials, key=lambda t: t.best_score):
                f.write(f"Trial {trial.trial_id}: {trial.best_score:.2f} dB ")
                f.write(f"({trial.status}) - {trial.iterations:,} iterations\n")

            # Elimination history
            f.write("\n" + self.elimination_strategy.get_elimination_report())

        self.log_callback(f"\nReport saved to: {report_path}")