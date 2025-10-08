"""Individual trial management for tournament-style pattern generation

Manages the state, performance tracking, and checkpointing for individual trials
in the tournament optimizer.
"""

import time
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta


@dataclass
class TrialState:
    """Complete state of a trial for checkpointing"""
    trial_id: int
    iteration: int
    patterns_complete: List[Any]
    current_pattern_id: int
    current_freq_sequence: np.ndarray
    current_iq_trajectory: np.ndarray
    current_lambda: float
    temperature: float
    best_cost: float
    random_state: Any
    score_history: List[float]
    timestamp: datetime


class Trial:
    """Manages a single optimization trial"""

    def __init__(self, trial_id: int, seed: int, p_cores: List[int] = None):
        self.trial_id = trial_id
        self.seed = seed
        self.p_cores = p_cores or []

        # State tracking
        self.status = 'pending'  # pending, running, paused, eliminated, completed
        self.iterations = 0
        self.patterns_complete = []
        self.patterns_file = None  # Path to saved patterns
        self.start_time = None
        self.end_time = None

        # Performance metrics
        # Start with inf, will be updated with actual scores from worker
        self.best_score = float('inf')  # Will be overwritten by actual scores
        self.current_score = float('inf')
        self.score_history = []
        self.convergence_rate = 0.001  # Small non-zero default
        self.improvement_history = []

        # Resource allocation
        self.compute_budget = 0
        self.bonus_budget = 0
        self.checkpoint_dir = Path(f"checkpoints/trial_{trial_id}")

        # UI display
        self.is_best = False
        self.eliminated = False
        self.progress = 0.0
        self.eta = "N/A"

        # Window orthogonality metrics (loaded from checkpoint)
        self.window_metrics = None
        self.global_metrics = None
        self.erasure_metrics = None
        self.weighted_score = None

    def start(self):
        """Start the trial"""
        self.status = 'running'
        self.start_time = datetime.now()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def pause(self):
        """Pause the trial"""
        if self.status == 'running':
            self.status = 'paused'
            self.save_checkpoint()

    def resume(self):
        """Resume the trial"""
        if self.status == 'paused':
            self.status = 'running'

    def eliminate(self, reason: str = ""):
        """Eliminate this trial from the tournament"""
        self.status = 'eliminated'
        self.eliminated = True
        self.end_time = datetime.now()
        self.save_checkpoint(final=True)

        if reason:
            with open(self.checkpoint_dir / "elimination.txt", "w") as f:
                f.write(f"Eliminated at iteration {self.iterations}\n")
                f.write(f"Reason: {reason}\n")
                f.write(f"Best score: {self.best_score:.3f} dB\n")

    def complete(self):
        """Mark trial as completed"""
        self.status = 'completed'
        self.end_time = datetime.now()
        self.save_checkpoint(final=True)

    def update_score(self, score: float):
        """Update trial score and metrics"""
        self.current_score = score
        self.score_history.append(score)

        # Update best score
        if score < self.best_score:
            # Only record improvement if we have a meaningful baseline
            if np.isfinite(self.best_score) and np.isfinite(score):
                improvement = self.best_score - score
                self.improvement_history.append((self.iterations, improvement))
            self.best_score = score

        # Calculate convergence rate (improvement over last 10 updates)
        if len(self.score_history) >= 10:
            recent_scores = self.score_history[-10:]
            older_scores = self.score_history[-20:-10] if len(self.score_history) >= 20 else self.score_history[:10]

            # Check for valid values
            if all(np.isfinite(recent_scores)) and all(np.isfinite(older_scores)):
                recent_avg = np.mean(recent_scores)
                older_avg = np.mean(older_scores)

                # Calculate rate of improvement
                score_diff = older_avg - recent_avg
                if np.isfinite(score_diff):
                    # Rate per update (not per iteration)
                    self.convergence_rate = abs(score_diff) / 10
                else:
                    self.convergence_rate = 0.001
            else:
                self.convergence_rate = 0.001
        else:
            self.convergence_rate = 0.001  # Default until we have enough history

    def calculate_eta(self, remaining_iterations: int):
        """Calculate estimated time to completion"""
        if self.start_time and self.iterations > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.iterations / elapsed  # iterations per second

            if rate > 0:
                remaining_seconds = remaining_iterations / rate
                eta_time = datetime.now() + timedelta(seconds=remaining_seconds)
                self.eta = eta_time.strftime("%H:%M")
            else:
                self.eta = "N/A"

    def project_final_score(self, target_iterations: int) -> float:
        """Project final score based on current trajectory"""
        if len(self.improvement_history) < 2:
            return self.best_score

        # Fit logarithmic curve to improvement history
        iterations = np.array([i for i, _ in self.improvement_history])
        scores = np.array([s for _, s in self.improvement_history])

        # Simple linear projection (could use more sophisticated model)
        if self.convergence_rate > 0:
            remaining = target_iterations - self.iterations
            projected_improvement = self.convergence_rate * remaining
            return self.best_score - projected_improvement
        else:
            return self.best_score

    def save_checkpoint(self, final: bool = False):
        """Save trial state to checkpoint file"""
        checkpoint_name = f"checkpoint_{self.iterations:07d}.pkl"
        if final:
            checkpoint_name = f"final_{self.iterations:07d}.pkl"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        state = TrialState(
            trial_id=self.trial_id,
            iteration=self.iterations,
            patterns_complete=self.patterns_complete,
            current_pattern_id=len(self.patterns_complete),
            current_freq_sequence=getattr(self, 'current_freq_sequence', None),
            current_iq_trajectory=getattr(self, 'current_iq_trajectory', None),
            current_lambda=getattr(self, 'current_lambda', 0.0),
            temperature=getattr(self, 'temperature', 1.0),
            best_cost=self.best_score,
            random_state=getattr(self, 'random_state', None),
            score_history=self.score_history[-1000:],  # Keep last 1000 scores
            timestamp=datetime.now()
        )

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)

        # Clean old checkpoints (keep every 10th and last 5)
        if not final:
            self._cleanup_old_checkpoints()

    def load_checkpoint(self, checkpoint_path: Path = None):
        """Load trial state from checkpoint"""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            if not checkpoints:
                return False
            checkpoint_path = checkpoints[-1]

        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)

        # Restore state
        self.iterations = state.iteration
        self.patterns_complete = state.patterns_complete
        self.best_score = state.best_cost
        self.score_history = state.score_history

        # Set attributes for optimizer
        self.current_freq_sequence = state.current_freq_sequence
        self.current_iq_trajectory = state.current_iq_trajectory
        self.current_lambda = state.current_lambda
        self.temperature = state.temperature
        self.random_state = state.random_state

        return True

    def _cleanup_old_checkpoints(self):
        """Keep only important checkpoints to save disk space"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))

        if len(checkpoints) <= 10:
            return

        # Keep every 10th checkpoint and last 5
        to_keep = set()

        # Keep every 10th
        for i in range(0, len(checkpoints), 10):
            to_keep.add(checkpoints[i])

        # Keep last 5
        for cp in checkpoints[-5:]:
            to_keep.add(cp)

        # Delete others
        for cp in checkpoints:
            if cp not in to_keep:
                cp.unlink()

    def get_status_dict(self) -> Dict[str, Any]:
        """Get trial status for UI display"""
        return {
            'id': self.trial_id,
            'status': self.status,
            'p_cores': self.p_cores,
            'iterations': self.iterations,
            'best_score': self.best_score,
            'convergence_rate': self.convergence_rate,
            'progress': self.progress,
            'eta': self.eta,
            'is_best': self.is_best,
            'eliminated': self.eliminated
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics for analysis"""
        runtime = None
        if self.start_time:
            end = self.end_time or datetime.now()
            runtime = (end - self.start_time).total_seconds()

        return {
            'trial_id': self.trial_id,
            'seed': self.seed,
            'status': self.status,
            'iterations': self.iterations,
            'best_score': self.best_score,
            'convergence_rate': self.convergence_rate,
            'improvement_count': len(self.improvement_history),
            'runtime_seconds': runtime,
            'compute_budget': self.compute_budget,
            'bonus_budget': self.bonus_budget,
            'final_projection': self.project_final_score(400000)
        }