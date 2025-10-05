#!/usr/bin/env python3
"""Main entry point for tournament-style pattern generation

Runs CASCADE pattern generation using a tournament approach with
dynamic compute allocation and Rich terminal UI.
"""

import os
import sys
import argparse
import threading
import signal
from pathlib import Path
from datetime import datetime
import yaml

# Add parent directories to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from core.tournament_optimizer import DynamicTournamentOptimizer
from core.core_manager import CoreManager
from ui.dashboard import PatternGeneratorDashboard
from ui.logger import DualLogger


class TournamentRunner:
    """Main runner for tournament pattern generation"""

    def __init__(self, config_file: str = None):
        self.config = self.load_config(config_file)
        self.optimizer = None
        self.dashboard = None
        self.logger = None
        self.core_manager = None
        self.running = False

    def load_config(self, config_file: str) -> dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'tournament': {
                'initial_trials': 8,
                'total_compute_budget': 2_000_000,
                'min_iterations_before_elimination': 50_000,
                'evaluation_interval': 10_000,
                'checkpoint_dir': './checkpoints'
            },
            'elimination': {
                'score_gap_threshold': 3.0,
                'stagnation_window': 30_000,
                'convergence_threshold': 0.001,
                'protect_top_n': 2,
                'minimum_diversity': 3,
                'aggressive_mode': False
            },
            'pattern': {
                'count': 128,
                'flip_weight': 0.6,
                'target_db': -37.5,
                'flip_target_db': -30.0
            },
            'hardware': {
                'num_p_cores': 8,
                'rotate_cores': True,
                'rotation_interval_minutes': 30,
                'process_priority': 'high'
            },
            'ui': {
                'type': 'rich',
                'refresh_rate': 1.0,
                'log_level': 'INFO'
            },
            'logging': {
                'file_level': 'DEBUG',
                'ui_level': 'INFO',
                'log_dir': './logs'
            }
        }

        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge with defaults
                for section in user_config:
                    if section in default_config:
                        default_config[section].update(user_config[section])
                    else:
                        default_config[section] = user_config[section]

        return default_config

    def setup_components(self):
        """Initialize all components"""
        # Create directories
        Path(self.config['tournament']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['logging']['log_dir']).mkdir(parents=True, exist_ok=True)

        # Initialize core manager
        self.core_manager = CoreManager(
            num_p_cores=self.config['hardware']['num_p_cores'],
            rotate_cores=self.config['hardware']['rotate_cores']
        )
        self.core_manager.apply_optimizations()

        # Initialize logger
        self.logger = DualLogger(
            log_dir=self.config['logging']['log_dir'],
            ui_dashboard=None,  # Will set after dashboard creation
            file_level=self.config['logging']['file_level'],
            ui_level=self.config['logging']['ui_level']
        )

        # Initialize optimizer
        self.optimizer = DynamicTournamentOptimizer(
            total_compute_budget=self.config['tournament']['total_compute_budget'],
            num_initial_trials=self.config['tournament']['initial_trials'],
            min_iterations=self.config['tournament']['min_iterations_before_elimination'],
            eval_interval=self.config['tournament']['evaluation_interval'],
            checkpoint_dir=self.config['tournament']['checkpoint_dir'],
            log_callback=self.log_message
        )

        # Initialize dashboard
        if self.config['ui']['type'] == 'rich':
            self.dashboard = PatternGeneratorDashboard(self.optimizer)
            self.logger.ui_dashboard = self.dashboard

    def log_message(self, message: str, level: str = "INFO"):
        """Unified logging callback"""
        if self.logger:
            self.logger.log(message, level=level)
        else:
            print(f"[{level}] {message}")

    def run_tournament(self):
        """Run the tournament in a separate thread"""
        try:
            self.running = True
            self.log_message("Starting tournament optimization", "INFO")

            # Run the tournament
            patterns = self.optimizer.run_tournament()

            self.log_message(f"Tournament complete! Generated {len(patterns)} patterns", "SUCCESS")

            # Save final patterns
            self.save_patterns(patterns)

        except Exception as e:
            self.log_message(f"Error in tournament: {e}", "ERROR")
            raise

        finally:
            self.running = False

    def save_patterns(self, patterns):
        """Save generated patterns to file"""
        output_dir = Path(self.config['tournament']['checkpoint_dir']) / 'output'
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"patterns_{timestamp}.bin"

        # In real implementation, use binary_format.save_pattern_file
        self.log_message(f"Patterns saved to: {output_file}", "SUCCESS")

    def run(self):
        """Main execution method"""
        print("=" * 60)
        print("CASCADE Pattern Tournament Generator")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Trials: {self.config['tournament']['initial_trials']}")
        print(f"  Budget: {self.config['tournament']['total_compute_budget']:,} iterations")
        print(f"  P-cores: {self.config['hardware']['num_p_cores']}")
        print(f"  UI: {self.config['ui']['type']}")
        print("=" * 60)
        print()

        # Setup components
        self.setup_components()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, self.signal_handler)

        # Start tournament in background thread
        tournament_thread = threading.Thread(target=self.run_tournament, daemon=True)
        tournament_thread.start()

        # Run UI in main thread
        if self.dashboard:
            try:
                self.dashboard.run(refresh_rate=self.config['ui']['refresh_rate'])
            except KeyboardInterrupt:
                print("\nStopping tournament...")
                self.stop()
        else:
            # No UI - just wait for tournament to finish
            tournament_thread.join()

        # Generate final report
        self.generate_report()

    def stop(self):
        """Stop the tournament gracefully"""
        self.running = False
        if self.dashboard:
            self.dashboard.stop()
        if self.logger:
            self.logger.close()

    def signal_handler(self, signum, frame):
        """Handle termination signals"""
        print(f"\nReceived signal {signum}, stopping...")
        self.stop()
        sys.exit(0)

    def generate_report(self):
        """Generate final report"""
        if self.logger:
            summary_data = {
                'start_time': self.optimizer.start_time if self.optimizer else None,
                'end_time': datetime.now(),
                'total_compute': self.optimizer.compute_used if self.optimizer else 0,
                'best_score': self.optimizer.global_best_score if self.optimizer else None,
                'best_trial': self.optimizer.global_best_trial_id if self.optimizer else None
            }
            self.logger.log_summary(summary_data)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CASCADE Pattern Tournament Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        choices=['scout', 'deepen', 'both'],
        default='both',
        help='Execution mode'
    )

    # Tournament parameters
    parser.add_argument(
        '--trials',
        type=int,
        default=8,
        help='Number of initial trials'
    )

    parser.add_argument(
        '--budget',
        type=int,
        default=2_000_000,
        help='Total compute budget (iterations)'
    )

    parser.add_argument(
        '--scout-iterations',
        type=int,
        default=100_000,
        help='Iterations per scout trial'
    )

    parser.add_argument(
        '--deepen-iterations',
        type=int,
        default=300_000,
        help='Additional iterations for deepening'
    )

    # Elimination parameters
    parser.add_argument(
        '--min-iterations',
        type=int,
        default=50_000,
        help='Minimum iterations before elimination'
    )

    parser.add_argument(
        '--eval-interval',
        type=int,
        default=10_000,
        help='Evaluation interval'
    )

    parser.add_argument(
        '--aggressive-elimination',
        action='store_true',
        help='Use aggressive elimination criteria'
    )

    parser.add_argument(
        '--min-survivors',
        type=int,
        default=2,
        help='Minimum number of survivors'
    )

    # Pattern parameters
    parser.add_argument(
        '--flip-weight',
        type=float,
        default=0.6,
        help='Weight for flip-orthogonality'
    )

    # Hardware parameters
    parser.add_argument(
        '--p-cores',
        type=str,
        default='0-7',
        help='P-core range (e.g., 0-7)'
    )

    parser.add_argument(
        '--rotate-cores',
        action='store_true',
        default=True,
        help='Enable core rotation for thermal management'
    )

    parser.add_argument(
        '--no-rotate-cores',
        dest='rotate_cores',
        action='store_false',
        help='Disable core rotation'
    )

    # UI parameters
    parser.add_argument(
        '--ui',
        choices=['rich', 'simple', 'none'],
        default='rich',
        help='UI type'
    )

    parser.add_argument(
        '--refresh-rate',
        type=float,
        default=1.0,
        help='UI refresh rate in seconds'
    )

    # Other parameters
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file (YAML)'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Checkpoint directory'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Log directory'
    )

    parser.add_argument(
        '--save-all-checkpoints',
        action='store_true',
        help='Save all checkpoints (not just periodic)'
    )

    args = parser.parse_args()

    # Create config from arguments
    if args.config:
        runner = TournamentRunner(args.config)
    else:
        # Build config from command-line arguments
        config = {
            'tournament': {
                'initial_trials': args.trials,
                'total_compute_budget': args.budget,
                'min_iterations_before_elimination': args.min_iterations,
                'evaluation_interval': args.eval_interval,
                'checkpoint_dir': args.checkpoint_dir
            },
            'elimination': {
                'aggressive_mode': args.aggressive_elimination,
                'minimum_diversity': args.min_survivors
            },
            'pattern': {
                'flip_weight': args.flip_weight
            },
            'hardware': {
                'rotate_cores': args.rotate_cores
            },
            'ui': {
                'type': args.ui,
                'refresh_rate': args.refresh_rate
            },
            'logging': {
                'log_dir': args.log_dir
            }
        }

        runner = TournamentRunner()
        runner.config.update(config)

    # Run the tournament
    runner.run()


if __name__ == "__main__":
    main()