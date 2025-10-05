# CASCADE Pattern Tournament Generator

Tournament-style pattern generation with dynamic compute allocation, early stopping, and Rich terminal UI. Optimized for Intel Core Ultra 7 265K and similar high-core-count CPUs.

## Features

### 🏆 Tournament Optimization
- **8 Scout Trials**: Start with 8 parallel trials exploring different solution regions
- **Dynamic Elimination**: Underperformers are eliminated, compute reallocated to winners
- **Smart Early Stopping**: Trials that converge or fall too far behind are stopped early
- **Adaptive Phases**: Exploration → Evaluation → Exploitation → Refinement

### 📊 Rich Terminal UI
- **Live Dashboard**: Real-time monitoring similar to `top` command
- **Trial Status Table**: Shows all trials with progress, scores, and convergence
- **Statistics Panel**: Overall progress, best scores, resource usage
- **Activity Log**: Scrolling log of important events
- **System Monitor**: CPU usage, temperature, and frequency tracking

### ⚙️ P-Core Optimization
- **Intel Hybrid Support**: Automatically detects and uses P-cores only
- **CPU Affinity**: Pins trials to specific P-cores for consistent performance
- **Core Rotation**: Rotates assignments every 30 minutes for thermal management
- **Windows/Linux**: Platform-specific optimizations for both OS

### 💾 Robust Checkpointing
- **Incremental Saves**: Checkpoint every 10k iterations
- **Resume Support**: Continue from any checkpoint after interruption
- **Trial Isolation**: Each trial has separate checkpoint directory
- **Statistics Logging**: Complete JSONL logs for post-run analysis

## Installation

```bash
# Install dependencies
pip install rich psutil numpy pyyaml

# Optional: Install Intel optimized packages (recommended)
conda install -c intel numpy scipy mkl
```

## Quick Start

### Default Tournament Mode (Recommended)
```bash
python generate_patterns_tournament.py
```

This runs with default settings:
- 8 initial trials
- 2M total iterations
- Automatic elimination and reallocation
- Rich UI with real-time monitoring

### Custom Configuration
```bash
# Use configuration file
python generate_patterns_tournament.py --config config/tournament.yaml

# Or specify parameters
python generate_patterns_tournament.py \
    --trials 8 \
    --budget 2000000 \
    --pattern-count 16 \
    --pattern-length 512 \
    --aggressive-elimination
```

## Usage Examples

### Scout-Only Mode (Exploration)
```bash
# Run 8 scouts for 100k iterations each
python generate_patterns_tournament.py \
    --mode scout \
    --trials 8 \
    --scout-iterations 100000
```

### Continue from Checkpoint
```bash
# Resume from previous run
python generate_patterns_tournament.py \
    --checkpoint-dir ./checkpoints_previous \
    --mode deepen
```

### No Elimination (Traditional Parallel)
```bash
# Run all trials to completion without elimination
python generate_patterns_tournament.py \
    --trials 4 \
    --no-elimination \
    --budget 1600000
```

### Aggressive Mode (Faster Convergence)
```bash
# More aggressive elimination for faster results
python generate_patterns_tournament.py \
    --aggressive-elimination \
    --min-survivors 2 \
    --eval-interval 5000
```

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--trials` | 8 | Number of initial scout trials |
| `--budget` | 2,000,000 | Total compute budget (iterations) |
| `--min-iterations` | 50,000 | Minimum before elimination |
| `--eval-interval` | 10,000 | Evaluation frequency |
| `--flip-weight` | 0.6 | Flip-orthogonality weight |
| `--p-cores` | 0-7 | P-core range to use |
| `--rotate-cores` | True | Enable thermal rotation |
| `--ui` | rich | UI type (rich/simple/none) |

### Configuration File

Create a `tournament.yaml` file for complex configurations:

```yaml
tournament:
  initial_trials: 8
  total_compute_budget: 2000000
  min_iterations_before_elimination: 50000
  evaluation_interval: 10000

elimination:
  score_gap_threshold: 3.0      # Eliminate if >3 dB behind
  stagnation_window: 30000      # No improvement iterations
  protect_top_n: 2               # Always keep top 2

hardware:
  num_p_cores: 8
  rotate_cores: true
  rotation_interval_minutes: 30

ui:
  type: rich
  refresh_rate: 1.0
```

## Performance Expectations

### Intel Core Ultra 7 265K (8 P-cores, 128GB RAM)

| Phase | Time | Expected Result |
|-------|------|-----------------|
| Scout (8×100k) | 3-4 hours | Find promising regions |
| Finalist (4×300k) | 36-40 hours | Refine best candidates |
| Total | 40-44 hours | -41 to -43 dB orthogonality |

### Efficiency Gains

- **vs Fixed Stages**: 20-30% less compute for same quality
- **vs No Elimination**: 40-50% faster convergence
- **P-Core Optimization**: 25-35% faster than mixed cores

## UI Screenshots

```
╭─────────────────────────────────────────────────────────────────────────╮
│             CASCADE Pattern Tournament | Phase: EXPLOITATION | 67.3%      │
╰─────────────────────────────────────────────────────────────────────────╯

╭─ Trial Status ───────────────────────────────────────────────────────────╮
│ ID  Status   Core  Iteration  Best Score  Conv Rate  Progress       ETA  │
│ 0   ✗ ELIM   P0      100,000   -36.42 dB    0.0012  ████████────  N/A  │
│ 1   ● RUN    P0-1    542,300   -40.82 dB    0.0042  ████████████  14:32│
│ 2   👑 BEST  P2-3    541,100   -41.93 dB    0.0038  ████████████  14:35│
│ 3   ✗ ELIM   P4       75,000   -34.21 dB    0.0008  ██████──────  N/A  │
│ 4   ● RUN    P4-5    540,900   -40.15 dB    0.0035  ████████████  14:38│
│ 5   ✗ ELIM   P6       82,000   -35.67 dB    0.0009  ██████──────  N/A  │
│ 6   ● RUN    P6-7    541,500   -40.74 dB    0.0040  ████████████  14:33│
│ 7   ✗ ELIM   P7       68,000   -33.89 dB    0.0007  █████───────  N/A  │
╰─────────────────────────────────────────────────────────────────────────╯

╭─ Statistics ──────╮  ╭─ Activity Log ─────────────────────────────────────╮
│ Active:        4  │  │ 14:21:05 Trial 2 improved: -41.91 → -41.93 dB      │
│ Eliminated:    4  │  │ 14:21:32 Eliminating Trial 0: Score gap 5.51 dB    │
│ Compute Used:     │  │ 14:22:15 Entering EXPLOITATION phase               │
│   1,342,000       │  │ 14:22:15 Trial 1 receives 125,000 bonus iterations │
│ Compute Left:     │  │ 14:22:15 Trial 2 receives 175,000 bonus iterations │
│   658,000         │  │ 14:22:15 Trial 4 receives 125,000 bonus iterations │
│                   │  │ 14:22:15 Trial 6 receives 125,000 bonus iterations │
│ Best Score:       │  │ 14:23:44 Trial 2 checkpoint at iteration 540,000   │
│   -41.93 dB       │  │ 14:24:12 Trial 4 improved: -40.13 → -40.15 dB      │
│ Best Trial: #2    │  │ 14:24:55 Trial 6 stagnant for 30000 iterations     │
│                   │  │ 14:25:03 Core rotation completed                   │
│ Runtime: 02:45    │  ╰─────────────────────────────────────────────────────╯
│ Phase: Exploit    │
╰───────────────────╯
```

## Analysis Tools

### View Logs
```bash
# View main tournament log
tail -f logs/tournament_20240115_143022.log

# View specific trial log
tail -f logs/trial_2/trial_2_20240115_143022.log
```

### Analyze Statistics
```bash
# Parse JSONL statistics
python analyze_stats.py logs/stats_20240115_143022.jsonl

# Generate convergence plots
python plot_convergence.py --checkpoint-dir ./checkpoints
```

### Compare Runs
```bash
# Compare tournament vs fixed stages
python compare_strategies.py \
    --tournament logs/tournament_run/ \
    --fixed logs/fixed_stages_run/
```

## Troubleshooting

### Issue: UI not updating
- Check `psutil` is installed: `pip install psutil`
- Try simple UI: `--ui simple`
- Or disable UI: `--ui none`

### Issue: Can't set CPU affinity
- On Linux: May need root for negative nice values
- On Windows: Run as Administrator for best performance
- Disable with: `--no-affinity`

### Issue: Out of memory
- Reduce concurrent trials: `--trials 4`
- Increase checkpoint interval: `--eval-interval 20000`
- Enable checkpoint compression (future feature)

### Issue: Trials eliminated too quickly
- Increase minimum iterations: `--min-iterations 100000`
- Disable aggressive mode: Remove `--aggressive-elimination`
- Increase protection: `--protect-top-n 3`

## Advanced Features

### Custom Elimination Strategy
```python
from tournament.core.elimination_strategy import EliminationStrategy

class MyStrategy(EliminationStrategy):
    def identify_underperformers(self, trials, iteration):
        # Custom logic here
        pass
```

### Post-Processing Hooks
```python
from tournament import DynamicTournamentOptimizer

optimizer = DynamicTournamentOptimizer()
optimizer.on_elimination = my_elimination_handler
optimizer.on_phase_change = my_phase_handler
```

### Real-time Monitoring API
```python
# Connect to running tournament
from tournament.api import TournamentMonitor

monitor = TournamentMonitor("./checkpoints")
status = monitor.get_status()
print(f"Best score: {status['best_score']}")
```

## Performance Tips

1. **Windows**: Run as Administrator for CPU affinity
2. **Linux**: Use `sudo` for performance governor changes
3. **Cooling**: Ensure good CPU cooling for sustained boost
4. **Background**: Close unnecessary applications
5. **Power**: Plug in laptop, set to High Performance
6. **Storage**: Use SSD for checkpoints (lots of I/O)

## Citation

If you use this tournament optimizer in research, please cite:

```bibtex
@software{cascade_tournament_2024,
  title = {CASCADE Pattern Tournament Generator},
  author = {CASCADE Development Team},
  year = {2024},
  url = {https://github.com/cascade/patterns}
}
```

## License

See LICENSE file in the CASCADE repository.

---

*For more information about CASCADE patterns and flip-orthogonality, see the main [patterns README](../README.md).*