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

# Import pattern generation functions from parent module
import sys
patterns_parent_dir = Path(__file__).parent.parent.parent.parent
if str(patterns_parent_dir) not in sys.path:
    sys.path.insert(0, str(patterns_parent_dir))

import patterns
from patterns import Pattern, generate_pattern_set
from patterns.optimizer import optimize_pattern_two_phase


def run_single_trial_worker(trial_id: int, iterations: int, seed: int,
                           checkpoint_dir: str, p_cores: List[int] = None) -> Dict[str, Any]:
    """
    Standalone function to run a single trial in a subprocess.
    This must be a module-level function to be picklable.
    """
    import psutil
    import os
    import numpy as np
    from pathlib import Path

    # This runs in a separate process
    # Set CPU affinity if on Windows/Linux
    try:
        p = psutil.Process()
        if p_cores:
            p.cpu_affinity(p_cores)
            p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
    except:
        pass  # Affinity setting failed, continue anyway

    # Simulate optimization (replace with actual pattern generation)
    np.random.seed(seed)
    best_score = float('inf')
    score_history = []

    for i in range(iterations // 100):  # Simplified simulation
        # In real implementation, this would call optimize_pattern_two_phase
        score = best_score - np.random.exponential(0.1)  # Simulate improvement
        if score < best_score:
            best_score = score

        score_history.append(best_score)

        # Save checkpoint periodically
        if (i * 100) % 10000 == 0 and checkpoint_dir:
            trial_checkpoint_dir = Path(checkpoint_dir) / f"trial_{trial_id}"
            trial_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # In real implementation, save actual checkpoint

    # Calculate convergence rate
    if len(score_history) > 10:
        recent_scores = score_history[-10:]
        convergence_rate = abs(recent_scores[-1] - recent_scores[0]) / 10
    else:
        convergence_rate = 0.01

    # Return results
    return {
        'trial_id': trial_id,
        'iterations_run': iterations,
        'best_score': best_score,
        'final_iteration': iterations,
        'convergence_rate': convergence_rate,
        'score_history': score_history[-100:],  # Last 100 scores
    }


class DynamicTournamentOptimizer:
    """Tournament optimizer with dynamic compute allocation"""

    def __init__(
        self,
        total_compute_budget: int = 2_000_000,
        num_initial_trials: int = 8,
        min_iterations: int = 50_000,
        eval_interval: int = 10_000,
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
            trial.compute_budget = self.total_budget // self.num_initial_trials
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

    def run_tournament(self) -> List[Pattern]:
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
                # Test if ProcessPoolExecutor works
                with ProcessPoolExecutor(max_workers=1) as test_executor:
                    test_future = test_executor.submit(lambda: 42)
                    test_future.result(timeout=2)
                executor_class = ProcessPoolExecutor
                self.log_callback("Using ProcessPoolExecutor for parallel trials")
            except Exception as e:
                try:
                    # Fall back to ThreadPoolExecutor
                    executor_class = ThreadPoolExecutor
                    self.log_callback(f"Process pool failed ({type(e).__name__}), using ThreadPoolExecutor")
                except:
                    executor_class = None
                    self.log_callback("Both process and thread pools failed, using sequential execution")

        # Run trials with selected executor
        if executor_class:
            try:
                with executor_class(max_workers=batch_size) as executor:
                    futures = {}

                    for trial_id in self.active_trials:
                        trial = self.trials[trial_id]

                        # Check if trial has budget left
                        if trial.iterations >= trial.compute_budget + trial.bonus_budget:
                            continue

                        # Submit trial for execution
                        future = executor.submit(
                            run_single_trial_worker,
                            trial_id,
                            min(self.eval_interval, trial.compute_budget + trial.bonus_budget - trial.iterations),
                            trial.seed,
                            str(self.checkpoint_dir),
                            trial.p_cores
                        )
                        futures[future] = trial_id

                    # Collect results
                    for future in as_completed(futures):
                        trial_id = futures[future]
                        try:
                            result = future.result(timeout=300)  # 5 minute timeout
                            self._process_trial_result(trial_id, result)
                        except Exception as e:
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

        # Update trial state
        trial.iterations = result['final_iteration']
        trial.best_score = result['best_score']
        trial.convergence_rate = result['convergence_rate']
        trial.score_history.extend(result['score_history'])

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
        trial.progress = trial.iterations / (trial.compute_budget + trial.bonus_budget)
        trial.calculate_eta(trial.compute_budget + trial.bonus_budget - trial.iterations)

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

    def _finalize_tournament(self) -> List[Pattern]:
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
        # In real implementation, load from checkpoint
        # For now, return empty list as placeholder
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