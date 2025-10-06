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
                           num_patterns: int = 16, pattern_length: int = 512) -> Dict[str, Any]:
    """
    Standalone function to run a single trial in a subprocess.
    This must be a module-level function to be picklable.

    Args:
        num_patterns: Number of patterns to generate (default 16)
        pattern_length: Symbols per pattern after expansion (default 512)
    """
    # Immediate debug output - use simple file operations to ensure it works
    import os
    import sys
    import traceback

    # Create debug file with absolute path and simple operations
    try:
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
        pattern_core_length = pattern_length // 4  # Core is 1/4 of full (4x redundancy)
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
                    pattern_set = checkpoint['pattern_set']
                    start_iteration = checkpoint['iteration']
                    best_score = checkpoint['best_score']
                    score_history = checkpoint['score_history']
                    # Will continue below with GA initialization
                else:
                    # Old format - convert to GA
                    pattern_set = checkpoint['pattern_set']
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
                f.write(f"  Core length: {pattern_core_length} bits (4x redundancy)\n")
                f.write(f"  Min symbols for decode: {min_symbols_needed} (37.5% erasure)\n")
                f.write(f"  Welch bound: {welch_bound_db:.2f} dB (theoretical limit)\n")
                f.flush()

            # First, create repetition map (same for all patterns)
            # This defines which symbols repeat (QR-like structure)
            # Use uint16 to support core lengths > 255
            repetition_map = np.zeros(pattern_length, dtype=np.uint16)

            # Create interleaved repetition: pattern_core_length positions, each repeated 4x
            for data_pos in range(pattern_core_length):
                repetition_map[data_pos * 4] = data_pos
                repetition_map[data_pos * 4 + 1] = data_pos
                repetition_map[data_pos * 4 + 2] = data_pos
                repetition_map[data_pos * 4 + 3] = data_pos

            # Shuffle to spread repetitions (burst error resistance)
            shuffle_groups = np.arange(pattern_core_length)
            np.random.shuffle(shuffle_groups)
            shuffled_map = np.zeros(pattern_length, dtype=np.uint16)  # uint16 for large patterns
            for idx, group in enumerate(shuffle_groups):
                shuffled_map[idx * 4:(idx + 1) * 4] = [group] * 4

            repetition_map = shuffled_map

            with open(debug_file_path, 'a') as f:
                f.write(f"Created repetition map: {pattern_core_length} positions × 4 repetitions\n")
                f.flush()

        # HELPER FUNCTIONS (defined at worker level so always available regardless of checkpoint)

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

        # Fitness evaluation function for a pattern set
        def evaluate_fitness(pattern_set_cores, use_sampling=True, sample_size=30):
            """Calculate fitness (worst-case orthogonality) for a pattern set

            Args:
                pattern_set_cores: List of 128-bit core patterns
                use_sampling: If True, sample random pairs (faster)
                sample_size: Number of pairs to sample (default 30 of 120)
            """
            worst_normal = -100.0
            worst_flip = -100.0
            worst_erasure = -100.0

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
                # Expand and convert to ±1
                pi_full = pattern_set_cores[i][repetition_map]
                pj_full = pattern_set_cores[j][repetition_map]
                pi = 2 * pi_full.astype(np.float32) - 1
                pj = 2 * pj_full.astype(np.float32) - 1

                # Normal correlation
                xcorr = np.correlate(pi, pj, mode='full')
                peak = np.max(np.abs(xcorr))
                corr_db = 20 * np.log10(peak / pattern_length + 1e-10)
                worst_normal = max(worst_normal, corr_db)

                # Flip correlation
                xcorr_f = np.correlate(pi, -pj, mode='full')
                peak_f = np.max(np.abs(xcorr_f))
                corr_f_db = 20 * np.log10(peak_f / pattern_length + 1e-10)
                worst_flip = max(worst_flip, corr_f_db)

                # Erasure (quick test)
                erasure_db = test_with_erasure(pi, pj, 0.375)
                worst_erasure = max(worst_erasure, erasure_db)

            # Return worst-case (higher/less negative is worse)
            return max(worst_normal, worst_flip, worst_erasure)

        # Continue with initialization or checkpoint resume
        if not checkpoint_file.exists():
            # GENETIC ALGORITHM SETUP
            # pattern_core_length, population_size, num_elites already defined above

            with open(debug_file_path, 'a') as f:
                f.write(f"Initializing genetic algorithm:\n")
                f.write(f"  Population size: {population_size} pattern sets\n")
                f.write(f"  Patterns per set: {num_patterns}\n")
                f.write(f"  Core bits per pattern: {pattern_core_length}\n")
                f.write(f"  Elites preserved: {num_elites}\n")
                f.flush()

            # Initialize population: 32 sets of 16 patterns (128-bit cores)
            population = []
            for set_idx in range(population_size):
                pattern_set_core = []
                for i in range(num_patterns):
                    pattern_seed = seed + set_idx * 1000 + i * 7919
                    np.random.seed(pattern_seed)
                    pattern_core = np.random.randint(0, 2, pattern_core_length, dtype=np.uint8)
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

            # Fitness evaluation function for a pattern set
            def evaluate_fitness(pattern_set_cores, use_sampling=True, sample_size=30):
                """Calculate fitness (worst-case orthogonality) for a pattern set

                Args:
                    pattern_set_cores: List of 128-bit core patterns
                    use_sampling: If True, sample random pairs (faster)
                    sample_size: Number of pairs to sample (default 30 of 120)
                """
                worst_normal = -100.0
                worst_flip = -100.0
                worst_erasure = -100.0

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
                    # Expand and convert to ±1
                    pi_full = pattern_set_cores[i][repetition_map]
                    pj_full = pattern_set_cores[j][repetition_map]
                    pi = 2 * pi_full.astype(np.float32) - 1
                    pj = 2 * pj_full.astype(np.float32) - 1

                    # Normal correlation
                    xcorr = np.correlate(pi, pj, mode='full')
                    peak = np.max(np.abs(xcorr))
                    corr_db = 20 * np.log10(peak / pattern_length + 1e-10)
                    worst_normal = max(worst_normal, corr_db)

                    # Flip correlation
                    xcorr_f = np.correlate(pi, -pj, mode='full')
                    peak_f = np.max(np.abs(xcorr_f))
                    corr_f_db = 20 * np.log10(peak_f / pattern_length + 1e-10)
                    worst_flip = max(worst_flip, corr_f_db)

                    # Erasure (quick test)
                    erasure_db = test_with_erasure(pi, pj, 0.375)
                    worst_erasure = max(worst_erasure, erasure_db)

                # Return worst-case (higher/less negative is worse)
                return max(worst_normal, worst_flip, worst_erasure)

            # Evaluate initial population (full evaluation for baseline)
            fitness_scores = []
            for set_idx, pattern_set_cores in enumerate(population):
                fitness = evaluate_fitness(pattern_set_cores, use_sampling=False)
                fitness_scores.append((fitness, set_idx))

            # Sort by fitness (lower/more negative is better)
            fitness_scores.sort(key=lambda x: x[0])
            best_score = fitness_scores[0][0]
            best_set_idx = fitness_scores[0][1]
            pattern_set = population[best_set_idx]  # Track best for checkpointing
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
        mutation_rate = 0.10  # 10% of 128 bits = ~13 bits per mutation
        crossover_rate = 0.7  # 70% of offspring use crossover

        # Calculate generations from iterations
        # Each generation evaluates population_size sets
        total_generations = iterations // population_size
        current_iteration = start_iteration
        current_generation = start_iteration // population_size

        with open(debug_file_path, 'a') as f:
            f.write(f"Genetic algorithm parameters:\n")
            f.write(f"  Total generations: {total_generations}\n")
            f.write(f"  Population size: {population_size}\n")
            f.write(f"  Mutation rate: {mutation_rate:.1%} ({int(mutation_rate * pattern_core_length)} bits)\n")
            f.write(f"  Crossover rate: {crossover_rate:.1%}\n")
            f.write(f"  Elites: {num_elites}\n")
            f.write(f"  Adaptive sampling: 30 pairs (75% speedup), full eval every 10th gen\n")
            f.write(f"  Expected speed: ~30 iter/s sampled, ~8 iter/s full\n")
            f.flush()

        import time
        start_time = time.time()
        last_log_time = start_time

        # Main genetic algorithm loop
        while current_generation < total_generations:
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
                        offspring[pattern_idx][flip_positions] = 1 - offspring[pattern_idx][flip_positions]

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

            # Track best
            current_best = fitness_scores[0][0]
            if current_best < best_score:
                best_score = current_best
                best_set_idx = fitness_scores[0][1]
                pattern_set = population[best_set_idx]

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

                with open(debug_file_path, 'a') as f:
                    f.write(f"Generation {current_generation}/{total_generations} [{eval_type}]:\n")
                    f.write(f"  Best: {best_fitness:.2f} dB, Median: {median_fitness:.2f} dB, "
                           f"Worst: {worst_fitness:.2f} dB\n")
                    f.write(f"  Overall best: {best_score:.2f} dB\n")
                    f.write(f"  Elapsed: {elapsed:.1f}s, Speed: {speed_str}\n")
                    f.flush()
                last_log_time = current_time

            # Save checkpoint every 1000 iterations for live updates
            if current_iteration % 1000 == 0 or current_generation >= total_generations:
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
    
        # Save final pattern set with repetition map
        final_patterns_file = trial_checkpoint_dir / f"final_patterns_{current_iteration}.pkl"

        # Create repetition_maps list for backward compatibility (all same)
        repetition_maps = [repetition_map.copy() for _ in range(len(pattern_set))]

        final_data = {
            'patterns': pattern_set,  # 16 × 128 core patterns
            'repetition_map': repetition_map,  # Single shared 512-element map
            'repetition_maps': repetition_maps,  # List format for compatibility
            'num_patterns': len(pattern_set),
            'pattern_core_length': len(pattern_set[0]) if pattern_set else 0,  # 128
            'pattern_full_length': 512,  # After expansion
            'unique_data_positions': 128,
            'redundancy_factor': 4,
            'best_score': best_score,
            'trial_id': trial_id,
            'seed': seed,
            'iterations': current_iteration,
            'algorithm': 'genetic'
        }
        with open(final_patterns_file, 'wb') as f:
            pickle.dump(final_data, f)

        total_elapsed = time.time() - start_time
        iterations_done = current_iteration - start_iteration
        avg_speed = iterations_done / total_elapsed if total_elapsed > 0 else float('inf')

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

        # Return results
        return {
            'trial_id': trial_id,
            'iterations_run': current_iteration - start_iteration,
            'best_score': best_score,
            'final_iteration': current_iteration,
            'convergence_rate': convergence_rate,
            'score_history': score_history[-100:],  # Last 100 scores
            'patterns_file': str(final_patterns_file)
        }

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
        total_compute_budget: int = 4_800_000,
        num_initial_trials: int = 8,
        min_iterations: int = 200_000,
        eval_interval: int = 50_000,
        checkpoint_dir: str = "./checkpoints",
        log_callback: Optional[Callable] = None,
        execution_mode: str = "auto",  # "process", "thread", "sequential", or "auto"
        num_patterns: int = 16,  # Number of patterns to generate
        pattern_length: int = 512  # Symbols per pattern (after 4x expansion)
    ):
        self.total_budget = total_compute_budget
        self.num_initial_trials = num_initial_trials
        self.min_iterations = min_iterations
        self.eval_interval = eval_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_callback = log_callback or print
        self.execution_mode = execution_mode
        self.num_patterns = num_patterns
        self.pattern_length = pattern_length

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.elimination_strategy = EliminationStrategy()
        self.trials: List[Trial] = []
        self.active_trials: List[int] = []

        # Tracking
        self.compute_used = 0
        self.global_best_score = float('inf')
        self.global_best_trial_id = None
        self.start_time = None
        self.current_phase = 'exploration'
        self.running = False
        self.monitor_thread = None

    def initialize_trials(self):
        """Initialize all trials"""
        self.trials = []
        for i in range(self.num_initial_trials):
            trial = Trial(
                trial_id=i,
                seed=1000 * (i + 1),
                p_cores=self._assign_p_cores(i)
            )
            # Ensure each trial gets at least min_iterations (50k)
            trial.compute_budget = max(
                self.min_iterations,
                self.total_budget // self.num_initial_trials
            )
            self.trials.append(trial)

        self.active_trials = list(range(self.num_initial_trials))
        self.log_callback(f"Initialized {self.num_initial_trials} trials")

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
        self.log_callback(f"Total compute budget: {self.total_budget:,} iterations")
        self.log_callback(f"Initial trials: {self.num_initial_trials}")
        self.log_callback("=" * 60)

        # Start background checkpoint monitoring for live updates
        self._start_checkpoint_monitor()

        # Main tournament loop
        while self.compute_used < self.total_budget and len(self.active_trials) > 0:
            # Update phase
            self._update_phase()

            # Run active trials for eval_interval
            self._run_trial_batch()

            # Evaluate and potentially eliminate
            # DISABLED for GA - elimination doesn't make sense with population-based optimization
            # Each trial runs its own population, all should complete
            # if self.compute_used >= self.min_iterations:
            #     self._evaluate_and_eliminate()

            # Check for convergence - DISABLED, always run to completion
            # if self._check_convergence():
            #     self.log_callback("Convergence achieved - stopping early")
            #     break

        # Stop monitoring
        self.running = False

        # Finalize and return best patterns
        return self._finalize_tournament()

    def _update_phase(self):
        """Update optimization phase based on progress"""
        progress = self.compute_used / self.total_budget

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
                    total_budget = trial.compute_budget + trial.bonus_budget
                    trial.progress = trial.iterations / total_budget if total_budget > 0 else 0
                    trial.calculate_eta(max(0, total_budget - trial.iterations))

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
            return

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

                        # Check if trial has budget left
                        if trial.iterations >= trial.compute_budget + trial.bonus_budget:
                            with open(master_debug, 'a') as f:
                                f.write(f"Skipping trial {trial_id} - budget exhausted\n")
                                f.flush()
                            continue

                        # Submit trial for execution
                        # First run: at least min_iterations (200k)
                        # Subsequent runs: eval_interval (50k) chunks
                        if trial.iterations == 0:
                            iterations_to_run = max(self.min_iterations, self.eval_interval)
                        else:
                            iterations_to_run = min(
                                self.eval_interval,
                                trial.compute_budget + trial.bonus_budget - trial.iterations
                            )

                        # Debug: Log what we're actually running
                        with open(master_debug, 'a') as f:
                            f.write(f"Trial {trial_id} scheduling:\n")
                            f.write(f"  trial.iterations: {trial.iterations}\n")
                            f.write(f"  min_iterations: {self.min_iterations}\n")
                            f.write(f"  eval_interval: {self.eval_interval}\n")
                            f.write(f"  iterations_to_run: {iterations_to_run}\n")
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
                            self.pattern_length
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

                # Check if trial has budget left
                if trial.iterations >= trial.compute_budget + trial.bonus_budget:
                    continue

                # Update trial status to running
                trial.status = 'running'
                trial.start()

                try:
                    # Run the trial directly (no parallelism)
                    # First run: at least min_iterations (50k)
                    # Subsequent runs: eval_interval (10k) chunks
                    if trial.iterations == 0:
                        iterations_to_run = max(self.min_iterations, self.eval_interval)
                    else:
                        iterations_to_run = min(
                            self.eval_interval,
                            trial.compute_budget + trial.bonus_budget - trial.iterations
                        )

                    result = run_single_trial_worker(
                        trial_id,
                        iterations_to_run,
                        trial.seed,
                        str(self.checkpoint_dir),
                        trial.p_cores,
                        self.num_patterns,
                        self.pattern_length
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

        # Update trial status based on budget
        total_budget = trial.compute_budget + trial.bonus_budget
        if trial.iterations >= total_budget:
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
        # This is always calculated, never incremented, to avoid double-counting
        self.compute_used = sum(t.iterations for t in self.trials)

        # Update progress
        trial.progress = trial.iterations / total_budget if total_budget > 0 else 1.0
        trial.calculate_eta(max(0, total_budget - trial.iterations))

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

            for trial_id, bonus in allocation.items():
                self.trials[trial_id].bonus_budget += bonus
                self.log_callback(f"Trial {trial_id} receives {bonus:,} bonus iterations")

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
                    # Handle both old format (just patterns) and new format (dict)
                    if isinstance(pattern_data, dict):
                        patterns = pattern_data.get('patterns', [])
                        repetition_maps = pattern_data.get('repetition_maps', [])
                        self.log_callback(f"\nLoaded {len(patterns)} patterns with repetition maps from best trial")
                        self.log_callback(f"Pattern length: {pattern_data.get('pattern_length', 'unknown')}")
                        self.log_callback(f"Unique data positions: {pattern_data.get('unique_data_positions', 'unknown')}")
                        return pattern_data
                    else:
                        # Old format - just patterns
                        self.log_callback(f"\nLoaded {len(pattern_data)} patterns (old format, no repetition maps)")
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