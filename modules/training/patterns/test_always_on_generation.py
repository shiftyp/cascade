#!/usr/bin/env python3
"""Test script for always-on center frequency signal generation with new patterns."""

import numpy as np
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from src.signal_generator.pattern_loader import PatternLoader
from src.signal_generator.gmsk import generate_gmsk_3fsk_always_on_center
import matplotlib.pyplot as plt


def test_always_on_generation():
    """Test the complete always-on signal generation pipeline."""

    print("="*60)
    print("Testing Always-On Signal Generation")
    print("="*60)

    # Initialize pattern loader in always-on mode
    # Specify the patterns/patterns directory where always_on subdirectory is
    patterns_dir = Path(__file__).parent / "patterns"
    loader = PatternLoader(patterns_dir=patterns_dir, use_always_on=True)

    # Test parameters
    pattern_id = 0
    pattern_length = 100  # Short pattern for testing

    # Define frequency triple (14 Hz spacing)
    tone_a_hz = 1300  # Lower
    tone_b_hz = 1314  # Center (14 Hz spacing)
    tone_c_hz = 1328  # Upper (14 Hz spacing)

    try:
        # Load the three pattern types
        print(f"\nLoading patterns for ID {pattern_id}...")
        center_pattern = loader.load_always_on_pattern(pattern_id, 'center', pattern_length)
        lower_pattern = loader.load_always_on_pattern(pattern_id, 'lower', pattern_length)
        upper_pattern = loader.load_always_on_pattern(pattern_id, 'upper', pattern_length)

        print(f"  Center pattern (first 20): {center_pattern[:20]}")
        print(f"  Lower pattern (first 20):  {lower_pattern[:20]}")
        print(f"  Upper pattern (first 20):   {upper_pattern[:20]}")

        # Verify pattern structure
        print("\nVerifying pattern structure...")

        # Check center has no gaps
        if np.any(center_pattern == -1):
            raise ValueError("Center pattern should not have gaps")
        print("  ✓ Center pattern is continuous")

        # Check lower is only active on even symbols
        for i in range(len(lower_pattern)):
            if i % 2 == 0 and lower_pattern[i] == -1:
                raise ValueError(f"Lower pattern missing at even index {i}")
            if i % 2 == 1 and lower_pattern[i] != -1:
                raise ValueError(f"Lower pattern active at odd index {i}")
        print("  ✓ Lower pattern active only on even symbols")

        # Check upper is only active on odd symbols
        for i in range(len(upper_pattern)):
            if i % 2 == 1 and upper_pattern[i] == -1:
                raise ValueError(f"Upper pattern missing at odd index {i}")
            if i % 2 == 0 and upper_pattern[i] != -1:
                raise ValueError(f"Upper pattern active at even index {i}")
        print("  ✓ Upper pattern active only on odd symbols")

        # Check data consistency
        for i in range(len(center_pattern)):
            if i % 2 == 0:  # Even
                if lower_pattern[i] != center_pattern[i]:
                    raise ValueError(f"Lower/center mismatch at index {i}")
            else:  # Odd
                if upper_pattern[i] != center_pattern[i]:
                    raise ValueError(f"Upper/center mismatch at index {i}")
        print("  ✓ Pattern data is consistent across frequencies")

        # Generate GMSK signal
        print("\nGenerating GMSK signal...")
        iq_signal = generate_gmsk_3fsk_always_on_center(
            center_pattern=center_pattern,
            lower_pattern=lower_pattern,
            upper_pattern=upper_pattern,
            tone_a_hz=tone_a_hz,
            tone_b_hz=tone_b_hz,
            tone_c_hz=tone_c_hz,
            sample_rate=48000,
            symbol_rate=200,
            num_centers=4  # Fixed 4-center configuration
        )

        print(f"  Generated signal shape: {iq_signal.shape}")
        print(f"  Signal power: {np.mean(np.abs(iq_signal)**2):.3f}")

        # Analyze spectrum
        print("\nAnalyzing spectrum...")
        from scipy import signal
        freqs, psd = signal.welch(iq_signal, fs=48000, nperseg=2048)

        # Find peak frequencies
        psd_db = 10 * np.log10(psd + 1e-12)
        peaks = []

        # Look for peaks near expected frequencies (with 4 centers)
        for center_idx in range(4):
            # Calculate offset for this center
            freq_offset = (center_idx - 2) * 60  # 60 Hz spacing between centers

            expected_freqs = [
                tone_a_hz + freq_offset,
                tone_b_hz + freq_offset,
                tone_c_hz + freq_offset
            ]

            for expected_freq in expected_freqs:
                # Find peak near expected frequency
                freq_window = (freqs > expected_freq - 10) & (freqs < expected_freq + 10)
                if np.any(freq_window):
                    peak_idx = np.argmax(psd_db[freq_window])
                    peak_freq = freqs[freq_window][peak_idx]
                    peak_power = psd_db[freq_window][peak_idx]
                    if peak_power > np.max(psd_db) - 30:  # Within 30 dB of max
                        peaks.append((peak_freq, peak_power))

        print(f"  Found {len(peaks)} significant peaks")
        for freq, power in sorted(peaks):
            print(f"    {freq:.1f} Hz: {power:.1f} dB")

        # Plot spectrum
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(freqs[freqs < 2000], psd_db[freqs < 2000])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.title('Always-On Signal Spectrum')
        plt.grid(True, alpha=0.3)

        # Mark expected frequencies
        for center_idx in range(4):
            freq_offset = (center_idx - 2) * 60
            plt.axvline(tone_a_hz + freq_offset, color='r', linestyle='--', alpha=0.5, label='Lower' if center_idx == 0 else '')
            plt.axvline(tone_b_hz + freq_offset, color='g', linestyle='--', alpha=0.5, label='Center' if center_idx == 0 else '')
            plt.axvline(tone_c_hz + freq_offset, color='b', linestyle='--', alpha=0.5, label='Upper' if center_idx == 0 else '')

        if len(peaks) > 0:
            plt.legend()

        # Plot time-domain signal
        plt.subplot(1, 2, 2)
        time_samples = min(1000, len(iq_signal))
        t = np.arange(time_samples) / 48000 * 1000  # ms
        plt.plot(t, np.real(iq_signal[:time_samples]), alpha=0.7, label='I')
        plt.plot(t, np.imag(iq_signal[:time_samples]), alpha=0.7, label='Q')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.title('Time-Domain Signal')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('always_on_test.png', dpi=100)
        print("\n  Saved plot to always_on_test.png")

        print("\n✅ Always-on signal generation test PASSED!")

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_always_on_generation()
    sys.exit(0 if success else 1)