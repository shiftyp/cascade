"""Pattern Visualization - IQ Trajectories, Lambda Distribution, Correlation Matrices

Generates visual analysis of pattern sets after generation.
"""

from typing import List
from pathlib import Path
import numpy as np
from .models import Pattern

# Lazy imports for matplotlib (not always available)
_MATPLOTLIB_AVAILABLE = None


def _check_matplotlib():
    """Check if matplotlib is available"""
    global _MATPLOTLIB_AVAILABLE
    if _MATPLOTLIB_AVAILABLE is None:
        try:
            import matplotlib
            import seaborn
            _MATPLOTLIB_AVAILABLE = True
        except ImportError:
            _MATPLOTLIB_AVAILABLE = False
    return _MATPLOTLIB_AVAILABLE


def plot_iq_trajectories(
    patterns: List[Pattern],
    output_file: str,
    sample_size: int = 20
):
    """Plot IQ trajectories in complex plane

    Args:
        patterns: List of patterns to visualize
        output_file: PNG file path
        sample_size: Number of patterns to plot (default 20 to avoid clutter)
    """
    if not _check_matplotlib():
        print("⚠ matplotlib not available, skipping IQ trajectory plot")
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))

    # Sample patterns to plot
    n_patterns = len(patterns)
    if n_patterns > sample_size:
        indices = np.linspace(0, n_patterns - 1, sample_size, dtype=int)
    else:
        indices = range(n_patterns)

    # Color by lambda complexity
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))

    for idx, color in zip(indices, colors):
        pattern = patterns[idx]
        iq = pattern.iq_trajectory

        # Plot trajectory
        ax.plot(iq.real, iq.imag, 'o-', alpha=0.5, color=color, markersize=3,
                label=f'P{pattern.pattern_id} (λ={pattern.iq_complexity_lambda:.2f})')

    ax.set_xlabel('I (In-phase)')
    ax.set_ylabel('Q (Quadrature)')
    ax.set_title(f'IQ Trajectories (Sample of {len(indices)} patterns)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # Add unit circle reference
    circle = plt.Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax.add_patch(circle)

    # Legend (show first 10 only to avoid clutter)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 10:
        ax.legend(handles[:10], labels[:10], loc='upper right', fontsize=8)
    else:
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"  ✓ Saved IQ trajectory plot: {output_file}")


def plot_frequency_heatmap(
    patterns: List[Pattern],
    output_file: str
):
    """Plot frequency sequence heatmap showing tone usage over time

    Args:
        patterns: List of patterns
        output_file: PNG file path
    """
    if not _check_matplotlib():
        print("⚠ matplotlib not available, skipping frequency heatmap")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create frequency matrix: patterns × time
    freq_matrix = np.array([p.freq_sequence for p in patterns])

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        freq_matrix,
        cmap='viridis',
        cbar_kws={'label': 'Tone Index (0-3)'},
        ax=ax,
        vmin=0,
        vmax=3
    )

    ax.set_xlabel('Time Symbol (0-31)')
    ax.set_ylabel('Pattern ID')
    ax.set_title(f'Frequency Sequence Heatmap ({len(patterns)} patterns)')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"  ✓ Saved frequency heatmap: {output_file}")


def plot_lambda_distribution(
    patterns: List[Pattern],
    output_file: str
):
    """Plot IQ complexity (λ) distribution histogram

    Args:
        patterns: List of patterns
        output_file: PNG file path
    """
    if not _check_matplotlib():
        print("⚠ matplotlib not available, skipping lambda distribution plot")
        return

    import matplotlib.pyplot as plt

    lambdas = [p.iq_complexity_lambda for p in patterns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(lambdas, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('IQ Complexity (λ)')
    ax1.set_ylabel('Number of Patterns')
    ax1.set_title('λ Distribution Histogram')
    ax1.axvline(np.mean(lambdas), color='red', linestyle='--', label=f'Mean: {np.mean(lambdas):.3f}')
    ax1.axvline(np.median(lambdas), color='orange', linestyle='--', label=f'Median: {np.median(lambdas):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_lambdas = np.sort(lambdas)
    cumulative = np.arange(1, len(lambdas) + 1) / len(lambdas) * 100

    ax2.plot(sorted_lambdas, cumulative, linewidth=2)
    ax2.set_xlabel('IQ Complexity (λ)')
    ax2.set_ylabel('Cumulative % of Patterns')
    ax2.set_title('Cumulative λ Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 100)

    # Add reference lines
    for pct in [25, 50, 75]:
        idx = int(len(lambdas) * pct / 100)
        if idx < len(sorted_lambdas):
            lambda_val = sorted_lambdas[idx]
            ax2.axhline(pct, color='gray', linestyle=':', alpha=0.5)
            ax2.text(0.05, pct + 2, f'{pct}%: λ={lambda_val:.2f}', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"  ✓ Saved lambda distribution: {output_file}")


def plot_correlation_matrix(
    patterns: List[Pattern],
    output_file: str,
    sample_size: int = 64
):
    """Plot correlation matrix heatmap

    Args:
        patterns: List of patterns
        output_file: PNG file path
        sample_size: Max patterns to include (default 64, full matrix too large)
    """
    if not _check_matplotlib():
        print("⚠ matplotlib not available, skipping correlation matrix")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns
    from .correlation import compute_4d_correlation

    n = min(len(patterns), sample_size)
    sampled_patterns = patterns[:n]

    # Compute correlation matrix
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            corr_db = compute_4d_correlation(sampled_patterns[i], sampled_patterns[j])
            corr_matrix[i, j] = corr_db
            corr_matrix[j, i] = corr_db  # Symmetric

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr_matrix,
        cmap='RdYlGn_r',  # Red for high correlation (bad), green for low (good)
        center=-37.5,  # Target threshold
        vmin=-50,
        vmax=-30,
        cbar_kws={'label': 'Correlation (dB)'},
        ax=ax,
        square=True
    )

    ax.set_xlabel('Pattern ID')
    ax.set_ylabel('Pattern ID')
    ax.set_title(f'Correlation Matrix (Sampled {n}/{len(patterns)} patterns)\nTarget: <-37.5 dB')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"  ✓ Saved correlation matrix: {output_file}")


def generate_batch_report(
    patterns: List[Pattern],
    batch_num: int,
    output_dir: str = "modules/training/data/visualizations"
):
    """Generate all visualization plots for a trial batch

    Args:
        patterns: List of patterns from best trial in batch
        batch_num: Batch number for file naming
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n  Generating visualizations for batch {batch_num}...")

    # Generate all plots
    plot_iq_trajectories(
        patterns,
        str(output_path / f"batch_{batch_num}_iq_trajectories.png")
    )

    plot_frequency_heatmap(
        patterns,
        str(output_path / f"batch_{batch_num}_frequency_heatmap.png")
    )

    plot_lambda_distribution(
        patterns,
        str(output_path / f"batch_{batch_num}_lambda_distribution.png")
    )

    plot_correlation_matrix(
        patterns,
        str(output_path / f"batch_{batch_num}_correlation_matrix.png")
    )

    print(f"  ✓ Batch {batch_num} visualizations complete\n")
