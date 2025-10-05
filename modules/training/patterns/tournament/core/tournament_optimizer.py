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
                           checkpoint_dir: str, p_cores: List[int] = None) -> Dict[str, Any]:
    """
    Standalone function to run a single trial in a subprocess.
    This must be a module-level function to be picklable.
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

        with open(debug_file_path, 'a') as f:
            f.write(f"Patterns dir: {patterns_dir}\n")
            f.write(f"sys.path before: {sys.path}\n")
            f.flush()

        if str(patterns_dir) not in sys.path:
            sys.path.insert(0, str(patterns_dir))

        with open(debug_file_path, 'a') as f:
            f.write(f"sys.path after: {sys.path}\n")
            f.write("About to import zadoff_chu...\n")
            f.flush()

        # Import pattern generation
        from zadoff_chu import generate_zadoff_chu_pattern

        with open(debug_file_path, 'a') as f:
            f.write("Successfully imported zadoff_chu\n")
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

        # Check for existing checkpoint
        checkpoint_file = trial_checkpoint_dir / "latest_checkpoint.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                pattern_set = checkpoint['pattern_set']
                start_iteration = checkpoint['iteration']
                best_score = checkpoint['best_score']
                score_history = checkpoint['score_history']
                temperature = checkpoint.get('temperature', 1.0)
        else:
            with open(debug_file_path, 'a') as f:
                f.write("No checkpoint found, generating initial patterns\n")
                f.flush()

            # Generate initial 128 frequency patterns (2-FSK binary sequences)
            pattern_set = []
            pattern_length = 32  # 32 symbols at 200 symbols/second = 160ms (CASCADE standard)

            with open(debug_file_path, 'a') as f:
                f.write(f"Generating {128} patterns of length {pattern_length}\n")
                f.flush()

            # Use Zadoff-Chu sequences as base for first 31 patterns
            for i in range(31):
                freq_seq = generate_zadoff_chu_pattern(u=i, N=pattern_length)
                pattern_set.append(freq_seq)

            # Random patterns for the rest
            for i in range(31, 128):
                freq_seq = np.random.randint(0, 2, pattern_length, dtype=np.uint8)
                pattern_set.append(freq_seq)

            with open(debug_file_path, 'a') as f:
                f.write(f"Generated {len(pattern_set)} patterns\n")
                f.write("Calculating initial correlations...\n")
                f.flush()

            # Calculate initial score (max cross-correlation across all pairs)
            max_correlation = -100.0
            correlation_count = 0

            for i in range(128):
                for j in range(i + 1, 128):
                    pattern_i = pattern_set[i].astype(np.float32) - 0.5
                    pattern_j = pattern_set[j].astype(np.float32) - 0.5

                    # Normal correlation
                    corr_normal = np.abs(np.dot(pattern_i, pattern_j))
                    corr_normal_db = 20 * np.log10(corr_normal / pattern_length + 1e-10)
                    max_correlation = max(max_correlation, corr_normal_db)

                    # Flip correlation (FSK inversion)
                    pattern_j_flip = -pattern_j
                    corr_flip = np.abs(np.dot(pattern_i, pattern_j_flip))
                    corr_flip_db = 20 * np.log10(corr_flip / pattern_length + 1e-10)
                    # Weight flip correlation (target -30dB vs -37.5dB)
                    max_correlation = max(max_correlation, corr_flip_db + 7.5)

                    correlation_count += 2

            best_score = max_correlation
            score_history = [best_score]
            start_iteration = 0
            temperature = 1.0

            with open(debug_file_path, 'a') as f:
                f.write(f"Initial best score: {best_score:.2f} dB\n")
                f.write(f"Starting optimization from iteration {start_iteration}\n")
                f.flush()
    
        # Simulated annealing parameters
        cooling_rate = 0.999
        min_temperature = 0.01
    
        # Run optimization iterations
        iterations_per_update = max(100, iterations // 100)  # Update at most 100 times
        current_iteration = start_iteration
        mutations_per_iteration = 10  # Test multiple mutations per iteration for better optimization

        with open(debug_file_path, 'a') as f:
            f.write(f"Starting optimization loop:\n")
            f.write(f"  Iterations per update: {iterations_per_update}\n")
            f.write(f"  Target iterations: {iterations}\n")
            f.write(f"  Mutations per iteration: {mutations_per_iteration}\n")
            f.flush()

        import time
        start_time = time.time()
        last_log_time = start_time

        while current_iteration < iterations:
            batch_iterations = min(iterations_per_update, iterations - current_iteration)

            # Optimize each pattern in the set
            for iter_count in range(batch_iterations):
                # Select random pattern to mutate
                pattern_idx = np.random.randint(0, 128)
                original_pattern = pattern_set[pattern_idx].copy()

                # Mutate pattern (flip random bits based on temperature)
                num_flips = max(1, min(10, int(temperature * 15)))  # 1-10 bits, more aggressive
                flip_positions = np.random.choice(len(original_pattern), num_flips, replace=False)
                mutated_pattern = original_pattern.copy()
                mutated_pattern[flip_positions] = 1 - mutated_pattern[flip_positions]

                # Calculate ONLY correlations involving the mutated pattern
                # This is much more targeted and likely to show improvement
                max_correlation_new = -100.0

                pattern_mut = mutated_pattern.astype(np.float32) - 0.5

                for j in range(128):
                    if j != pattern_idx:
                        pattern_j = pattern_set[j].astype(np.float32) - 0.5

                        # Normal correlation
                        corr_normal = np.abs(np.dot(pattern_mut, pattern_j))
                        corr_normal_db = 20 * np.log10(corr_normal / len(pattern_mut) + 1e-10)
                        max_correlation_new = max(max_correlation_new, corr_normal_db)

                        # Flip correlation
                        pattern_j_flip = -pattern_j
                        corr_flip = np.abs(np.dot(pattern_mut, pattern_j_flip))
                        corr_flip_db = 20 * np.log10(corr_flip / len(pattern_mut) + 1e-10)
                        weighted_flip_db = corr_flip_db + 7.5
                        max_correlation_new = max(max_correlation_new, weighted_flip_db)

                # For comparison, calculate current pattern's worst correlation
                max_correlation_old = -100.0
                pattern_old = original_pattern.astype(np.float32) - 0.5

                for j in range(128):
                    if j != pattern_idx:
                        pattern_j = pattern_set[j].astype(np.float32) - 0.5

                        # Normal correlation
                        corr_normal = np.abs(np.dot(pattern_old, pattern_j))
                        corr_normal_db = 20 * np.log10(corr_normal / len(pattern_old) + 1e-10)
                        max_correlation_old = max(max_correlation_old, corr_normal_db)

                        # Flip correlation
                        pattern_j_flip = -pattern_j
                        corr_flip = np.abs(np.dot(pattern_old, pattern_j_flip))
                        corr_flip_db = 20 * np.log10(corr_flip / len(pattern_old) + 1e-10)
                        weighted_flip_db = corr_flip_db + 7.5
                        max_correlation_old = max(max_correlation_old, weighted_flip_db)

                # Decide whether to keep mutation
                # Accept if the mutated pattern has lower worst-case correlation than original
                if max_correlation_new < max_correlation_old:
                    # Improvement! Keep the mutation
                    pattern_set[pattern_idx] = mutated_pattern

                    # Update global best score periodically (every 100 iterations)
                    # by checking the actual worst case across ALL patterns
                    if current_iteration % 100 == 0:
                        actual_worst = -100.0
                        for i in range(128):
                            for j in range(i + 1, 128):
                                pi = pattern_set[i].astype(np.float32) - 0.5
                                pj = pattern_set[j].astype(np.float32) - 0.5

                                # Normal
                                corr = np.abs(np.dot(pi, pj))
                                corr_db = 20 * np.log10(corr / len(pi) + 1e-10)
                                actual_worst = max(actual_worst, corr_db)

                                # Flip
                                corr_f = np.abs(np.dot(pi, -pj))
                                corr_f_db = 20 * np.log10(corr_f / len(pi) + 1e-10)
                                actual_worst = max(actual_worst, corr_f_db + 7.5)

                        best_score = actual_worst

                elif temperature > min_temperature:
                    # Sometimes accept worse solutions based on temperature
                    delta = max_correlation_new - max_correlation_old
                    probability = np.exp(-delta / temperature)
                    if np.random.random() < probability:
                        pattern_set[pattern_idx] = mutated_pattern

                # Cool down
                temperature = max(min_temperature, temperature * cooling_rate)
                current_iteration += 1

                if current_iteration >= iterations:
                    break
    
            # Update score history
            score_history.append(best_score)

            # Log progress periodically (every 10 seconds or every 1000 iterations)
            current_time = time.time()
            if current_iteration % 1000 == 0 or (current_time - last_log_time) >= 10:
                elapsed = current_time - start_time
                if elapsed > 0:
                    iter_per_sec = (current_iteration - start_iteration) / elapsed
                    speed_str = f"{iter_per_sec:.1f} iter/s"
                else:
                    speed_str = "N/A"

                with open(debug_file_path, 'a') as f:
                    f.write(f"Progress: iteration {current_iteration}/{iterations}, "
                           f"best_score={best_score:.2f} dB, "
                           f"elapsed={elapsed:.1f}s, "
                           f"speed={speed_str}\n")
                    f.flush()
                last_log_time = current_time

            # Save checkpoint periodically
            if current_iteration % 10000 == 0 or current_iteration >= iterations:
                checkpoint = {
                    'pattern_set': pattern_set,
                    'iteration': current_iteration,
                    'best_score': best_score,
                    'score_history': score_history,
                    'temperature': temperature,
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
    
        # Save final pattern set
        final_patterns_file = trial_checkpoint_dir / f"final_patterns_{current_iteration}.pkl"
        with open(final_patterns_file, 'wb') as f:
            pickle.dump(pattern_set, f)

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
        total_compute_budget: int = 2_000_000,
        num_initial_trials: int = 8,
        min_iterations: int = 50_000,
        eval_interval: int = 5_000,  # Reduced since iterations are now slower
        checkpoint_dir: str = "./checkpoints",
        log_callback: Optional[Callable] = None,
        execution_mode: str = "auto"  # "process", "thread", "sequential", or "auto"
    ):
        self.total_budget = total_compute_budget
        self.num_initial_trials = num_initial_trials
        self.min_iterations = min_iterations
        self.eval_interval = eval_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_callback = log_callback or print
        self.execution_mode = execution_mode

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

    def run_tournament(self) -> List[np.ndarray]:
        """Run the tournament optimization"""
        self.start_time = datetime.now()
        self.initialize_trials()

        self.log_callback("=" * 60)
        self.log_callback("Starting CASCADE Pattern Tournament")
        self.log_callback(f"Total compute budget: {self.total_budget:,} iterations")
        self.log_callback(f"Initial trials: {self.num_initial_trials}")
        self.log_callback("=" * 60)

        # Main tournament loop
        while self.compute_used < self.total_budget and len(self.active_trials) > 0:
            # Update phase
            self._update_phase()

            # Run active trials for eval_interval
            self._run_trial_batch()

            # Evaluate and potentially eliminate
            if self.compute_used >= self.min_iterations:
                self._evaluate_and_eliminate()

            # Check for convergence
            if self._check_convergence():
                self.log_callback("Convergence achieved - stopping early")
                break

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
                        iterations_to_run = min(self.eval_interval, trial.compute_budget + trial.bonus_budget - trial.iterations)

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
                            trial.p_cores
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
                    result = run_single_trial_worker(
                        trial_id,
                        min(self.eval_interval, trial.compute_budget + trial.bonus_budget - trial.iterations),
                        trial.seed,
                        str(self.checkpoint_dir),
                        trial.p_cores
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

        # Update compute used
        self.compute_used += result['iterations_run']

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
        """Check if tournament has converged"""
        if len(self.active_trials) <= 1:
            return True  # Only one trial left

        # Check if all trials have stagnated
        active_trial_objects = [t for t in self.trials if t.trial_id in self.active_trials]
        all_stagnant = all(t.convergence_rate < 0.0001 for t in active_trial_objects)

        if all_stagnant and self.compute_used > self.total_budget * 0.5:
            return True

        # Check if best score is exceptional
        if self.global_best_score < -45.0:  # Exceptional result
            return True

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

        # Load and return best patterns
        if hasattr(best_trial, 'patterns_file') and best_trial.patterns_file:
            import pickle
            try:
                with open(best_trial.patterns_file, 'rb') as f:
                    pattern_set = pickle.load(f)
                    self.log_callback(f"\nLoaded {len(pattern_set)} patterns from best trial")
                    return pattern_set
            except Exception as e:
                self.log_callback(f"Error loading patterns: {e}")
                return []
        else:
            # Try to load from checkpoint directory
            checkpoint_dir = self.checkpoint_dir / f"trial_{best_trial.trial_id}"
            pattern_files = sorted(checkpoint_dir.glob("final_patterns_*.pkl"))
            if pattern_files:
                import pickle
                with open(pattern_files[-1], 'rb') as f:
                    pattern_set = pickle.load(f)
                    self.log_callback(f"\nLoaded {len(pattern_set)} patterns from checkpoint")
                    return pattern_set
            else:
                self.log_callback("No patterns file found")
                return []

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