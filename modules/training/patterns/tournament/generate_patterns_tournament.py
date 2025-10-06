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

    def load_config(self, config_file: str = None) -> dict:
        """Load configuration from file or use defaults"""
        # Check for default config file if none specified
        if not config_file:
            default_path = Path(__file__).parent / 'config' / 'tournament.yaml'
            if default_path.exists():
                config_file = str(default_path)
                # Don't print during init to avoid corrupting Rich UI

        default_config = {
            'tournament': {
                'initial_trials': 8,
                'total_compute_budget': 4_800_000,
                'evaluation_interval': 50_000,
                'checkpoint_dir': './checkpoints'
            },
            'elimination': {
                'score_gap_threshold': 3.0,
                'stagnation_window': 100_000,
                'convergence_threshold': 0.001,
                'protect_top_n': 2,
                'minimum_diversity': 3,
                'aggressive_mode': False
            },
            'pattern': {
                'count': 16,
                'length': 512,
                'unique_data_positions': 128,
                'redundancy_factor': 4,
                'erasure_tolerance': 0.375,
                'target_normal_db': -30.0,
                'target_flip_db': -28.0,
                'target_erasure_db': -27.0
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
            # Don't print during loading to avoid corrupting Rich UI
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge with defaults
                for section in user_config:
                    if section in default_config:
                        default_config[section].update(user_config[section])
                    else:
                        default_config[section] = user_config[section]

                # Store loaded config info for later logging
                default_config['_config_file'] = config_file
                if 'tournament' in user_config:
                    budget = user_config['tournament'].get('total_compute_budget')
                    if budget:
                        default_config['_loaded_budget'] = budget
        elif config_file:
            # Store warning for later display
            default_config['_config_warning'] = f"Config file not found: {config_file}"

        return default_config

    def setup_components(self):
        """Initialize all components"""
        # Create directories with safe access
        tournament_config = self.config.get('tournament', {})
        logging_config = self.config.get('logging', {})
        pattern_config = self.config.get('pattern', {})

        # Partition checkpoints and logs by pattern configuration
        num_patterns = pattern_config.get('count', 16)
        pattern_length = pattern_config.get('length', 512)
        config_suffix = f"p{num_patterns}_l{pattern_length}"

        base_checkpoint_dir = tournament_config.get('checkpoint_dir', './checkpoints')
        base_log_dir = logging_config.get('log_dir', './logs')

        checkpoint_dir = f"{base_checkpoint_dir}/{config_suffix}"
        log_dir = f"{base_log_dir}/{config_suffix}"

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Store partitioned paths for later use
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.config_suffix = config_suffix  # Save for logging

        # Log partition info (if not using Rich UI, print it)
        if self.config.get('ui', {}).get('type', 'rich') != 'rich':
            print(f"Configuration: {num_patterns} patterns × {pattern_length} symbols")
            print(f"Checkpoints: {checkpoint_dir}")
            print(f"Logs: {log_dir}")

        # Initialize core manager
        hardware_config = self.config.get('hardware', {})
        self.core_manager = CoreManager(
            num_p_cores=hardware_config.get('num_p_cores', 8),
            rotate_cores=hardware_config.get('rotate_cores', True)
        )
        self.core_manager.apply_optimizations()

        # Initialize logger (pass UI type to control console output)
        ui_type = self.config.get('ui', {}).get('type', 'rich')
        self.logger = DualLogger(
            log_dir=log_dir,
            ui_dashboard=None,  # Will set after dashboard creation
            file_level=logging_config.get('file_level', 'DEBUG'),
            ui_level=logging_config.get('ui_level', 'INFO'),
            use_console=(ui_type != 'rich')  # Disable console output for Rich UI
        )

        # Initialize optimizer with partitioned checkpoint_dir
        pattern_config = self.config.get('pattern', {})

        # Handle generations parameter (preferred) or compute_budget (legacy)
        if '_generations' in tournament_config:
            total_generations = tournament_config['_generations']
        elif 'total_generations' in tournament_config:
            total_generations = tournament_config['total_generations']
        else:
            # Legacy: convert budget to generations
            total_budget = tournament_config.get('total_compute_budget', 4_800_000)
            total_generations = total_budget // 32  # population_size = 32

        self.optimizer = DynamicTournamentOptimizer(
            total_generations=total_generations,
            num_initial_trials=tournament_config.get('initial_trials', 8),
            checkpoint_dir=checkpoint_dir,  # Already partitioned above
            log_callback=self.log_message,
            execution_mode=hardware_config.get('execution_mode', 'auto'),
            num_patterns=pattern_config.get('count', 16),
            pattern_length=pattern_config.get('length', 512)
        )

        # Initialize dashboard
        ui_config = self.config.get('ui', {})
        if ui_config.get('type', 'rich') == 'rich':
            self.dashboard = PatternGeneratorDashboard(self.optimizer)
            self.logger.ui_dashboard = self.dashboard

    def log_message(self, message: str, level: str = "INFO"):
        """Unified logging callback"""
        if self.logger:
            self.logger.log(message, level=level)
        elif self.config.get('ui', {}).get('type', 'rich') != 'rich':
            # Only print if not using Rich UI
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

    def save_patterns(self, pattern_data):
        """Save generated patterns to file"""
        # Use the partitioned checkpoint_dir
        output_dir = Path(self.checkpoint_dir) / 'output'
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include configuration in filename for clarity
        pickle_file = output_dir / f"patterns_{self.config_suffix}_{timestamp}.pkl"
        import pickle
        with open(pickle_file, 'wb') as f:
            pickle.dump(pattern_data, f)

        self.log_message(f"Patterns saved to: {pickle_file}", "SUCCESS")

        # Log summary of what was saved
        if isinstance(pattern_data, dict):
            patterns = pattern_data.get('patterns', [])
            repetition_maps = pattern_data.get('repetition_maps', [])
            self.log_message(f"  {len(patterns)} patterns", "INFO")
            self.log_message(f"  Pattern length: {pattern_data.get('pattern_length', 'unknown')}", "INFO")
            self.log_message(f"  Unique data positions: {pattern_data.get('unique_data_positions', 'unknown')}", "INFO")
            self.log_message(f"  Redundancy factor: {pattern_data.get('redundancy_factor', 'unknown')}x", "INFO")

        # TODO: In real implementation, also convert to binary_format for CASCADE modem

    def run(self):
        """Main execution method"""
        # Only print to console if not using Rich UI
        ui_type = self.config.get('ui', {}).get('type', 'rich')
        if ui_type != 'rich':
            print("=" * 60)
            print("CASCADE Pattern Tournament Generator")
            print("=" * 60)
            print(f"Configuration:")
            print(f"  Trials: {self.config.get('tournament', {}).get('initial_trials', 8)}")
            print(f"  Budget: {self.config.get('tournament', {}).get('total_compute_budget', 2_000_000):,} iterations")
            print(f"  P-cores: {self.config.get('hardware', {}).get('num_p_cores', 8)}")
            print(f"  UI: {ui_type}")
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
                refresh_rate = self.config.get('ui', {}).get('refresh_rate', 1.0)
                self.dashboard.run(refresh_rate=refresh_rate)
            except KeyboardInterrupt:
                # Don't print when using Rich UI - it will handle the display
                if self.config.get('ui', {}).get('type', 'rich') != 'rich':
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
        # Don't print when using Rich UI
        if self.config.get('ui', {}).get('type', 'rich') != 'rich':
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
        default=4_800_000,
        help='Total compute budget (iterations). For GA: iterations = generations × population_size(32)'
    )

    parser.add_argument(
        '--generations',
        type=int,
        help='Total generations for GA (alternative to --budget). If specified, overrides --budget.'
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

    # Elimination parameters (DISABLED - GA runs all trials to completion)
    # Kept for backward compatibility but ignored

    # Pattern parameters
    parser.add_argument(
        '--pattern-count',
        type=int,
        default=16,
        help='Number of patterns to generate'
    )

    parser.add_argument(
        '--pattern-length',
        type=int,
        default=512,
        help='Symbols per pattern'
    )

    # Hardware parameters
    parser.add_argument(
        '--p-cores',
        type=str,
        default='0-7',
        help='P-core range (e.g., 0-7)'
    )

    parser.add_argument(
        '--execution-mode',
        choices=['auto', 'process', 'thread', 'sequential'],
        default='auto',
        help='Execution mode: auto (default), process (multiprocessing), thread (threading), or sequential'
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

    # Always create runner (loads default config if available)
    runner = TournamentRunner(args.config)

    # Override with command-line arguments only if they differ from defaults
    # This allows the YAML file to take precedence over hardcoded defaults
    if not args.config:
        # Parse P-cores argument
        p_cores = args.p_cores
        if '-' in p_cores:
            start, end = map(int, p_cores.split('-'))
            num_p_cores = end - start + 1
        else:
            num_p_cores = int(p_cores)

        # Only override if values were explicitly provided (differ from defaults)
        if args.trials != 8:
            runner.config.setdefault('tournament', {})['initial_trials'] = args.trials

        # Handle --generations parameter (converts to iterations)
        if args.generations:
            # For GA: iterations = generations × population_size
            # Population size is hardcoded to 32 in worker
            iterations = args.generations * 32
            runner.config.setdefault('tournament', {})['total_compute_budget'] = iterations
            runner.config.setdefault('tournament', {})['_generations'] = args.generations
        elif args.budget != 4800000:
            runner.config.setdefault('tournament', {})['total_compute_budget'] = args.budget

        if args.checkpoint_dir != './checkpoints':
            runner.config.setdefault('tournament', {})['checkpoint_dir'] = args.checkpoint_dir

        if args.pattern_count != 16:
            runner.config.setdefault('pattern', {})['count'] = args.pattern_count

        if args.pattern_length != 512:
            runner.config.setdefault('pattern', {})['length'] = args.pattern_length

        # Always set hardware config from command line
        runner.config.setdefault('hardware', {})['num_p_cores'] = num_p_cores
        runner.config.setdefault('hardware', {})['rotate_cores'] = args.rotate_cores
        runner.config.setdefault('hardware', {})['execution_mode'] = args.execution_mode

        if args.ui != 'rich':
            runner.config.setdefault('ui', {})['type'] = args.ui

        if args.refresh_rate != 1.0:
            runner.config.setdefault('ui', {})['refresh_rate'] = args.refresh_rate

        if args.log_dir != './logs':
            runner.config.setdefault('logging', {})['log_dir'] = args.log_dir

    # Run the tournament
    runner.run()


if __name__ == "__main__":
    main()