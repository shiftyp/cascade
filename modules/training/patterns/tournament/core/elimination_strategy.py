"""Smart elimination strategies for tournament optimization

Implements various criteria for identifying underperforming trials
while protecting diversity and avoiding premature elimination.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class EliminationConfig:
    """Configuration for elimination strategy"""
    score_gap_threshold: float = 3.0  # dB behind leader
    stagnation_window: int = 30000  # iterations without improvement
    convergence_threshold: float = 0.001  # minimum convergence rate
    protect_top_n: int = 2  # always keep top N trials
    minimum_diversity: int = 3  # minimum trials to maintain
    min_iterations_before_elimination: int = 50000
    aggressive_mode: bool = False


class EliminationStrategy:
    """Manages trial elimination decisions"""

    def __init__(self, config: EliminationConfig = None):
        self.config = config or EliminationConfig()
        self.elimination_history = []

    def identify_underperformers(
        self,
        trials: List,
        iteration: int
    ) -> List[Tuple[int, str]]:
        """Identify trials to eliminate with reasons

        Args:
            trials: List of Trial objects
            iteration: Current global iteration count

        Returns:
            List of (trial_id, reason) tuples for elimination
        """
        if iteration < self.config.min_iterations_before_elimination:
            return []  # Too early to eliminate

        active_trials = [t for t in trials if t.status == 'running']

        if len(active_trials) <= self.config.minimum_diversity:
            return []  # Maintain minimum diversity

        # Gather metrics
        scores = [t.best_score for t in active_trials]
        conv_rates = [t.convergence_rate for t in active_trials]
        trial_ids = [t.trial_id for t in active_trials]

        # Calculate statistics
        best_score = min(scores)
        median_score = np.median(scores)
        score_std = np.std(scores)

        # Sort trials by score (best first)
        sorted_indices = np.argsort(scores)

        to_eliminate = []

        for idx, trial in enumerate(active_trials):
            rank = sorted_indices.tolist().index(idx)

            # Never eliminate top performers
            if rank < self.config.protect_top_n:
                continue

            eliminate = False
            reason = ""

            # Criterion 1: Far behind leader
            score_gap = trial.best_score - best_score
            if score_gap > self.config.score_gap_threshold:
                if iteration > 100000 or self.config.aggressive_mode:
                    eliminate = True
                    reason = f"Score gap {score_gap:.2f} dB behind leader"

            # Criterion 2: Stagnation
            if not eliminate and self._is_stagnant(trial):
                if trial.best_score > median_score:
                    eliminate = True
                    reason = f"Stagnant for {self.config.stagnation_window} iterations"

            # Criterion 3: Poor convergence rate
            if not eliminate and trial.convergence_rate < self.config.convergence_threshold:
                if trial.best_score > median_score and iteration > 75000:
                    eliminate = True
                    reason = f"Poor convergence rate ({trial.convergence_rate:.4f})"

            # Criterion 4: Statistical projection
            if not eliminate and iteration > 150000:
                projected_final = trial.project_final_score(400000)
                if projected_final > best_score + 2.0:
                    eliminate = True
                    reason = f"Projected to remain {projected_final - best_score:.2f} dB behind"

            # Protection: High convergence rate (still improving fast)
            if eliminate and trial.convergence_rate > max(conv_rates) * 0.8:
                eliminate = False  # Keep fast improvers

            # Protection: Diversity check
            if eliminate and self._would_reduce_diversity(trials, trial, to_eliminate):
                eliminate = False  # Maintain pattern diversity

            if eliminate:
                to_eliminate.append((trial.trial_id, reason))

        # Final check: Don't eliminate too many at once
        max_eliminations = max(1, len(active_trials) // 4)
        if len(to_eliminate) > max_eliminations:
            # Keep only the worst performers
            to_eliminate = sorted(
                to_eliminate,
                key=lambda x: next(t.best_score for t in trials if t.trial_id == x[0]),
                reverse=True
            )[:max_eliminations]

        # Record elimination decisions
        for trial_id, reason in to_eliminate:
            self.elimination_history.append({
                'iteration': iteration,
                'trial_id': trial_id,
                'reason': reason,
                'score': next(t.best_score for t in trials if t.trial_id == trial_id)
            })

        return to_eliminate

    def _is_stagnant(self, trial) -> bool:
        """Check if a trial is stagnant"""
        if len(trial.improvement_history) < 2:
            return False

        # Check last improvement
        if trial.improvement_history:
            last_improvement_iteration = trial.improvement_history[-1][0]
            iterations_since_improvement = trial.iterations - last_improvement_iteration

            if iterations_since_improvement > self.config.stagnation_window:
                return True

        # Also check if score history is flat
        if len(trial.score_history) >= 20:
            recent = trial.score_history[-10:]
            older = trial.score_history[-20:-10]

            if np.std(recent) < 0.01 and abs(np.mean(recent) - np.mean(older)) < 0.01:
                return True

        return False

    def _would_reduce_diversity(self, trials, trial_to_eliminate, already_eliminating) -> bool:
        """Check if eliminating this trial would reduce solution diversity

        This is a simplified check - a real implementation might analyze
        the actual patterns for diversity.
        """
        # Count remaining trials after elimination
        remaining_count = len([t for t in trials if t.status == 'running'])
        remaining_count -= len(already_eliminating)
        remaining_count -= 1  # This trial

        if remaining_count < self.config.minimum_diversity:
            return True

        # Could add more sophisticated diversity metrics here
        # For example, checking if patterns are exploring different solution regions

        return False

    def should_eliminate_early(self, trial, global_best_score: float, iteration: int) -> bool:
        """Quick check for very poor performers that should be eliminated early"""
        if iteration < 25000:
            return False  # Give everyone a fair start

        # Very far behind (>5 dB) and not improving
        if trial.best_score > global_best_score + 5.0 and trial.convergence_rate < 0.0001:
            return True

        # Stuck at a very poor score
        if trial.best_score > -30.0 and iteration > 50000:
            return True

        return False

    def compute_reallocation(
        self,
        eliminated_trials: List[int],
        surviving_trials: List,
        remaining_budget: int
    ) -> Dict[int, int]:
        """Compute how to reallocate compute from eliminated trials

        Args:
            eliminated_trials: List of eliminated trial IDs
            surviving_trials: List of surviving Trial objects
            remaining_budget: Total compute budget remaining

        Returns:
            Dict mapping trial_id to additional iterations
        """
        if not surviving_trials:
            return {}

        # Calculate freed compute
        freed_compute = len(eliminated_trials) * (remaining_budget // (len(eliminated_trials) + len(surviving_trials)))

        # Allocate based on performance
        scores = [t.best_score for t in surviving_trials]

        # Weight allocation by inverse score (better trials get more)
        weights = 1.0 / (np.array(scores) - min(scores) + 1.0)
        weights = weights / weights.sum()

        allocation = {}
        for trial, weight in zip(surviving_trials, weights):
            additional = int(freed_compute * weight)
            allocation[trial.trial_id] = additional
            trial.bonus_budget += additional

        return allocation

    def get_elimination_report(self) -> str:
        """Generate a report of elimination history"""
        if not self.elimination_history:
            return "No eliminations yet"

        report = "Elimination History:\n"
        report += "=" * 60 + "\n"

        for record in self.elimination_history:
            report += f"Iteration {record['iteration']:,}: "
            report += f"Trial {record['trial_id']} "
            report += f"(score: {record['score']:.2f} dB)\n"
            report += f"  Reason: {record['reason']}\n"

        return report

    def adjust_aggressiveness(self, phase: str):
        """Adjust elimination criteria based on optimization phase"""
        if phase == 'exploration':
            # Early phase - be lenient
            self.config.score_gap_threshold = 5.0
            self.config.stagnation_window = 50000
            self.config.protect_top_n = 3

        elif phase == 'evaluation':
            # Mid phase - standard criteria
            self.config.score_gap_threshold = 3.0
            self.config.stagnation_window = 30000
            self.config.protect_top_n = 2

        elif phase == 'exploitation':
            # Late phase - be aggressive
            self.config.score_gap_threshold = 2.0
            self.config.stagnation_window = 20000
            self.config.protect_top_n = 1

        elif phase == 'refinement':
            # Final phase - keep only the best
            self.config.score_gap_threshold = 1.0
            self.config.stagnation_window = 10000
            self.config.protect_top_n = 1