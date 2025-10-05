#!/usr/bin/env python3
"""Quick script to check optimization progress from debug logs"""

import sys
from pathlib import Path
import re

def check_progress(trial_id=0):
    """Check progress from debug log"""
    log_file = Path(f'logs/debug_trial_{trial_id}.txt')

    if not log_file.exists():
        print(f"No debug log found for trial {trial_id}")
        return

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Find all progress lines
    progress_lines = []
    for line in lines:
        if 'Progress:' in line:
            # Extract iteration and score
            match = re.search(r'iteration (\d+)/\d+.*best_score=([-\d.]+)', line)
            if match:
                iteration = int(match.group(1))
                score = float(match.group(2))
                progress_lines.append((iteration, score))

    if not progress_lines:
        print(f"No progress found in trial {trial_id} log")
        return

    print(f"\nTrial {trial_id} Progress:")
    print("-" * 40)
    print(f"Initial score: {progress_lines[0][1]:.2f} dB")
    print(f"Current score: {progress_lines[-1][1]:.2f} dB")
    print(f"Improvement: {progress_lines[0][1] - progress_lines[-1][1]:.2f} dB")
    print(f"Iterations: {progress_lines[-1][0]}")

    # Show recent progress
    print("\nRecent checkpoints:")
    for iter, score in progress_lines[-5:]:
        print(f"  Iter {iter:6d}: {score:6.2f} dB")

    # Check if improving
    if len(progress_lines) > 1:
        recent_improvement = progress_lines[-2][1] - progress_lines[-1][1]
        if recent_improvement > 0:
            print(f"\n✓ Score IS improving (recent: {recent_improvement:.3f} dB better)")
        elif recent_improvement < 0:
            print(f"\n✗ Score got worse (recent: {-recent_improvement:.3f} dB worse)")
        else:
            print("\n- Score unchanged in recent checkpoint")

if __name__ == "__main__":
    trial_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    check_progress(trial_id)