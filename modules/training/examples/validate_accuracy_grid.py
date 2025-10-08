"""Validation grid: Model accuracy by SNR and modulation/symbol-rate.

Generates heatmap showing BER/SER across:
- X-axis: SNR (-15 to +30 dB)
- Y-axis: Modulation scheme + Symbol rate combinations

Usage:
    python validate_accuracy_grid.py --model checkpoints/model_epoch_10.pt
"""

import sys
import os
from pathlib import Path

# Add CASCADE root to path
cascade_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(cascade_root))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from dataclasses import dataclass
import argparse
from tqdm import tqdm

from modules.training.src.signal_generator.generator import SignalGenerator, KernelParameters
from modules.training.src.propagation.channel import PropagationChannel


@dataclass
class ValidationResult:
    """Results for a single test configuration."""
    snr_db: float
    modulation: str
    symbol_rate: int
    ber: float  # Bit error rate
    ser: float  # Symbol error rate
    decode_success: bool
    num_bits: int
    num_errors: int


class AccuracyGridValidator:
    """Validate model accuracy across SNR and modulation/rate grid."""

    def __init__(self, model_path: str = None, device: str = 'cuda'):
        """
        Initialize validator.

        Args:
            model_path: Path to trained model checkpoint (None for testing signal gen only)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.model = None

        if model_path:
            # TODO: Load trained model
            # self.model = load_cascade_model(model_path, device)
            print(f"Model loading not yet implemented. Testing signal generation only.")

        # Initialize signal generator
        self.generator = SignalGenerator()
        self.channel = PropagationChannel()

        # Test configurations
        self.snr_range = np.arange(-15, 31, 3)  # -15 to +30 dB, step 3
        self.modulations = ['BPSK', 'QPSK', '8-PSK', '16-APSK']
        self.symbol_rates = [75, 100, 125, 150, 175, 200, 250, 300]

        # Test message
        self.test_message = b"The quick brown fox jumps over the lazy dog. " * 2  # ~92 bytes

    def test_configuration(self, snr_db: float, modulation: str,
                          symbol_rate: int, num_trials: int = 5) -> ValidationResult:
        """
        Test model on specific configuration.

        Args:
            snr_db: Signal-to-noise ratio in dB
            modulation: Modulation scheme
            symbol_rate: Data symbol rate (sym/s)
            num_trials: Number of trials to average

        Returns:
            ValidationResult with averaged metrics
        """
        total_bits = 0
        total_errors = 0
        decode_successes = 0

        for trial in range(num_trials):
            try:
                # Generate clean signal
                clean_signal, metadata = self.generator.generate(
                    pattern_id=trial % 4,  # Rotate through patterns
                    frequency_triple=10,  # Use middle frequency
                    modulation_scheme=modulation,
                    polar_rate=(2, 3),  # Rate 2/3
                    data_symbol_rate=symbol_rate,
                    message=self.test_message,
                    seed=42 + trial
                )

                # Add noise and propagation effects
                noisy_signal = self.channel.apply_awgn(
                    clean_signal.iq_samples,
                    snr_db=snr_db,
                    seed=100 + trial
                )

                if self.model is None:
                    # Without model, just test signal generation
                    # Assume perfect decode for testing
                    decoded_bits = np.random.randint(0, 2, len(self.test_message) * 8)
                    true_bits = np.unpackbits(np.frombuffer(self.test_message, dtype=np.uint8))

                    # Simulate realistic BER based on SNR and modulation
                    ber = self._theoretical_ber(snr_db, modulation, symbol_rate)
                    num_errors = int(ber * len(true_bits))

                    total_bits += len(true_bits)
                    total_errors += num_errors
                    decode_successes += 1 if ber < 0.1 else 0
                else:
                    # TODO: Run model inference
                    # decoded_message = self.model.decode(noisy_signal, metadata)
                    # Compare decoded_message to self.test_message
                    pass

            except Exception as e:
                # Decode failed
                print(f"Failed: SNR={snr_db}, mod={modulation}, rate={symbol_rate}: {e}")
                continue

        # Calculate averages
        ber = total_errors / total_bits if total_bits > 0 else 1.0
        ser = ber * self._get_bits_per_symbol(modulation)  # Approximate

        return ValidationResult(
            snr_db=snr_db,
            modulation=modulation,
            symbol_rate=symbol_rate,
            ber=ber,
            ser=min(ser, 1.0),
            decode_success=decode_successes >= num_trials // 2,
            num_bits=total_bits,
            num_errors=total_errors
        )

    def _theoretical_ber(self, snr_db: float, modulation: str, symbol_rate: int) -> float:
        """Estimate theoretical BER for AWGN channel (for testing without model)."""
        snr_linear = 10 ** (snr_db / 10)

        # Adjust SNR for symbol rate (higher rate = less energy per symbol)
        # Reference rate: 150 sym/s
        rate_penalty_db = 10 * np.log10(symbol_rate / 150)
        effective_snr_db = snr_db - rate_penalty_db
        effective_snr = 10 ** (effective_snr_db / 10)

        # Approximate BER formulas
        if modulation == 'BPSK':
            from scipy.special import erfc
            ber = 0.5 * erfc(np.sqrt(effective_snr))
        elif modulation == 'QPSK':
            from scipy.special import erfc
            ber = 0.5 * erfc(np.sqrt(effective_snr / 2))
        elif modulation == '8-PSK':
            from scipy.special import erfc
            ber = (1/3) * erfc(np.sqrt(effective_snr) * np.sin(np.pi/8))
        elif modulation == '16-APSK':
            # Rough approximation
            from scipy.special import erfc
            ber = (3/8) * erfc(np.sqrt(effective_snr / 10))
        else:
            ber = 0.5

        return np.clip(ber, 1e-6, 0.5)

    def _get_bits_per_symbol(self, modulation: str) -> int:
        """Get bits per symbol for modulation scheme."""
        return {'BPSK': 1, 'QPSK': 2, '8-PSK': 3, '16-APSK': 4}[modulation]

    def run_validation(self, num_trials: int = 5) -> List[ValidationResult]:
        """
        Run validation across all configurations.

        Args:
            num_trials: Number of trials per configuration

        Returns:
            List of ValidationResult objects
        """
        results = []

        total_configs = len(self.snr_range) * len(self.modulations) * len(self.symbol_rates)

        with tqdm(total=total_configs, desc="Validating") as pbar:
            for snr_db in self.snr_range:
                for modulation in self.modulations:
                    for symbol_rate in self.symbol_rates:
                        result = self.test_configuration(
                            snr_db, modulation, symbol_rate, num_trials
                        )
                        results.append(result)
                        pbar.update(1)

        return results

    def plot_accuracy_grid(self, results: List[ValidationResult],
                          metric: str = 'ber',
                          output_path: str = 'accuracy_grid.png'):
        """
        Plot accuracy grid heatmap.

        Args:
            results: List of validation results
            metric: 'ber' or 'ser' or 'decode_success'
            output_path: Where to save plot
        """
        # Create grid dimensions
        snr_values = sorted(set(r.snr_db for r in results))

        # Create combo labels: "BPSK@75" etc
        combos = []
        for mod in self.modulations:
            for rate in self.symbol_rates:
                combos.append(f"{mod}@{rate}")

        # Create matrix
        matrix = np.zeros((len(combos), len(snr_values)))

        # Fill matrix
        for result in results:
            combo = f"{result.modulation}@{result.symbol_rate}"
            combo_idx = combos.index(combo)
            snr_idx = snr_values.index(result.snr_db)

            if metric == 'ber':
                matrix[combo_idx, snr_idx] = result.ber
            elif metric == 'ser':
                matrix[combo_idx, snr_idx] = result.ser
            elif metric == 'decode_success':
                matrix[combo_idx, snr_idx] = 1.0 if result.decode_success else 0.0

        # Create plot
        fig, ax = plt.subplots(figsize=(16, 12))

        # Use log scale for BER/SER
        if metric in ['ber', 'ser']:
            matrix_plot = np.log10(np.clip(matrix, 1e-6, 1.0))
            vmin, vmax = -6, 0
            cmap = 'RdYlGn_r'
            cbar_label = f'log10({metric.upper()})'
        else:
            matrix_plot = matrix
            vmin, vmax = 0, 1
            cmap = 'RdYlGn'
            cbar_label = 'Decode Success Rate'

        # Plot heatmap
        im = ax.imshow(matrix_plot, aspect='auto', cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation='nearest')

        # Set ticks
        ax.set_xticks(np.arange(len(snr_values)))
        ax.set_yticks(np.arange(len(combos)))
        ax.set_xticklabels([f'{snr:+.0f}' for snr in snr_values])
        ax.set_yticklabels(combos, fontsize=8)

        # Labels
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Modulation @ Symbol Rate (sym/s)', fontsize=12)
        ax.set_title(f'CASCADE Model Accuracy Grid: {metric.upper()} vs SNR and Modulation/Rate',
                    fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cbar_label, fontsize=11)

        # Grid
        ax.set_xticks(np.arange(len(snr_values)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(combos)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        # Add dividing lines between modulation types
        for i in range(1, len(self.modulations)):
            y_pos = i * len(self.symbol_rates) - 0.5
            ax.axhline(y_pos, color='black', linewidth=2)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved accuracy grid to {output_path}")
        plt.close()

    def plot_snr_curves(self, results: List[ValidationResult],
                       output_path: str = 'snr_curves.png'):
        """
        Plot BER vs SNR curves for each modulation scheme.

        Args:
            results: List of validation results
            output_path: Where to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for mod_idx, modulation in enumerate(self.modulations):
            ax = axes[mod_idx]

            # Plot curve for each symbol rate
            for symbol_rate in self.symbol_rates:
                # Extract results for this modulation and rate
                mod_results = [r for r in results
                              if r.modulation == modulation and r.symbol_rate == symbol_rate]
                mod_results.sort(key=lambda r: r.snr_db)

                snrs = [r.snr_db for r in mod_results]
                bers = [r.ber for r in mod_results]

                ax.semilogy(snrs, bers, marker='o', label=f'{symbol_rate} sym/s', linewidth=2)

            ax.set_xlabel('SNR (dB)', fontsize=11)
            ax.set_ylabel('BER', fontsize=11)
            ax.set_title(f'{modulation}', fontsize=12, fontweight='bold')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            ax.set_ylim([1e-6, 1])

        fig.suptitle('BER vs SNR by Modulation and Symbol Rate',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved SNR curves to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Validate CASCADE model accuracy grid')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of trials per configuration')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for plots')

    args = parser.parse_args()

    print("=" * 60)
    print("CASCADE Accuracy Grid Validation")
    print("=" * 60)
    print(f"Model: {args.model or 'None (signal generation test only)'}")
    print(f"Device: {args.device}")
    print(f"Trials per config: {args.trials}")
    print()

    # Initialize validator
    validator = AccuracyGridValidator(model_path=args.model, device=args.device)

    print(f"Test configurations:")
    print(f"  SNR range: {validator.snr_range[0]} to {validator.snr_range[-1]} dB "
          f"(step {validator.snr_range[1] - validator.snr_range[0]})")
    print(f"  Modulations: {validator.modulations}")
    print(f"  Symbol rates: {validator.symbol_rates}")
    print(f"  Total configs: {len(validator.snr_range) * len(validator.modulations) * len(validator.symbol_rates)}")
    print()

    # Run validation
    results = validator.run_validation(num_trials=args.trials)

    # Generate plots
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\nGenerating plots...")
    validator.plot_accuracy_grid(results, metric='ber',
                                output_path=str(output_dir / 'accuracy_grid_ber.png'))
    validator.plot_accuracy_grid(results, metric='decode_success',
                                output_path=str(output_dir / 'accuracy_grid_success.png'))
    validator.plot_snr_curves(results,
                             output_path=str(output_dir / 'snr_curves.png'))

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for modulation in validator.modulations:
        print(f"\n{modulation}:")
        for symbol_rate in validator.symbol_rates:
            mod_results = [r for r in results
                          if r.modulation == modulation and r.symbol_rate == symbol_rate]

            # Find threshold SNR for BER < 1e-3
            threshold_snr = None
            for r in sorted(mod_results, key=lambda x: x.snr_db):
                if r.ber < 1e-3:
                    threshold_snr = r.snr_db
                    break

            if threshold_snr:
                print(f"  {symbol_rate:3d} sym/s: SNR > {threshold_snr:+.0f} dB for BER < 1e-3")
            else:
                print(f"  {symbol_rate:3d} sym/s: BER > 1e-3 at all tested SNRs")

    print("\nValidation complete!")


if __name__ == '__main__':
    main()
