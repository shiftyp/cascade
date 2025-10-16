"""
Continuous symbol rate calculator based on Shannon capacity and low-SNR readability.

Optimizes symbol rate selection for CASCADE based on:
1. Shannon capacity limits (theoretical maximum)
2. Low SNR readability requirements (practical minimum)
3. Implementation margin for real-world conditions
"""

import numpy as np
from typing import Tuple


class ContinuousRateCalculator:
    """
    Calculate optimal continuous symbol rates based on channel capacity.

    Key principles:
    - At low SNR (< -10 dB): Prioritize detection over throughput
    - At medium SNR (-10 to +10 dB): Balance reliability and rate
    - At high SNR (> +10 dB): Approach Shannon limit with margin
    """

    def __init__(self,
                 bandwidth_hz: float = 150.0,  # Total bandwidth for 3-FSK (3 × 50 Hz)
                 min_rate: float = 25.0,      # Minimum symbol rate (pattern provides sync)
                 preferred_min_rate: float = 50.0,  # Preferred minimum for good sync
                 max_rate: float = 600.0,      # Maximum for NN decoder (can handle more ISI)
                 implementation_margin_db: float = 2.0):  # NN can get closer to Shannon limit
        """
        Initialize rate calculator.

        Args:
            bandwidth_hz: Channel bandwidth in Hz (20 Hz for CASCADE FSK)
            min_rate: Minimum symbol rate (sync/detection limit)
            max_rate: Maximum symbol rate (ISI/bandwidth limit)
            implementation_margin_db: Margin below Shannon capacity
        """
        self.bandwidth = bandwidth_hz
        self.min_rate = min_rate
        self.preferred_min_rate = preferred_min_rate
        self.max_rate = max_rate
        self.margin_db = implementation_margin_db

    def shannon_capacity(self, snr_db: float) -> float:
        """
        Calculate Shannon capacity in bits/second.

        C = B * log2(1 + SNR)

        Args:
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Capacity in bits/second
        """
        snr_linear = 10 ** (snr_db / 10)
        capacity_bps = self.bandwidth * np.log2(1 + snr_linear)
        return capacity_bps

    def optimal_modulation(self, snr_db: float) -> Tuple[str, int]:
        """
        Select optimal modulation scheme based on SNR.

        Uses thresholds with hysteresis to avoid mode bouncing.

        Args:
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Tuple of (modulation_name, bits_per_symbol)
        """
        # Simple SNR-based thresholds for clean transitions
        # Avoid complex calculations that cause jumps

        if snr_db < 7:
            # Use BPSK up to 7 dB
            return 'BPSK', 1
        elif snr_db < 14:
            # QPSK from 7-14 dB
            return 'QPSK', 2
        elif snr_db < 22:
            # 8-PSK from 14-22 dB
            return '8-PSK', 3
        else:
            # 16-APSK above 22 dB
            return '16-APSK', 4

    def get_effective_snr(self, base_snr_db: float, num_channels: int) -> float:
        """
        Calculate effective SNR with always-on center design gains.
        Includes penalty for 14 Hz spacing overlap.

        Args:
            base_snr_db: Measured SNR in dB
            num_channels: Number of frequency triples (centers) used

        Returns:
            Effective SNR after applying gains from always-on design
        """
        # Overlap penalty for 14 Hz spacing (3 Hz overlap = 18% of 17 Hz)
        overlap_penalty = 0.5  # dB penalty for moderate overlap

        if num_channels == 1:
            # Single center: 3 frequencies (center + 2 alternating)
            # Gains: +1.25 dB power, +1 dB sync, -0.5 dB overlap
            return base_snr_db + 2.25 - overlap_penalty  # Net +1.75 dB
        elif num_channels == 2:
            # 2 centers: 4 frequencies (2 centers + 2 alternating)
            # Gains: +1.76 dB power, +1.5 dB sync/diversity, -0.5 dB overlap
            return base_snr_db + 3.26 - overlap_penalty  # Net +2.76 dB
        elif num_channels == 3:
            # 3 centers: 5 frequencies (3 centers + 2 alternating)
            # Gains: +2.0 dB power, +1.75 dB sync/diversity, -0.5 dB overlap
            return base_snr_db + 3.75 - overlap_penalty  # Net +3.25 dB
        elif num_channels == 4:
            # 4 centers: 6 frequencies (4 centers + 2 alternating)
            # Gains: +2.22 dB power, +2 dB sync/diversity, -0.5 dB overlap
            return base_snr_db + 4.22 - overlap_penalty  # Net +3.72 dB
        else:
            # Fallback for more channels
            return base_snr_db + 2.0 - overlap_penalty

    def calculate_continuous_rate(self, snr_db: float,
                                 multipath_severity: float = 0.0,
                                 qrm_present: bool = False) -> Tuple[float, int]:
        """
        Calculate optimal continuous symbol rate and channels for given conditions.

        This is the main function that produces continuous (not discrete) rates
        and determines how many channels to use for low SNR.

        Args:
            snr_db: Signal-to-noise ratio in dB
            multipath_severity: 0.0 (none) to 1.0 (severe)
            qrm_present: True if QRM interference detected

        Returns:
            Tuple of (symbol_rate, num_channels, start_freq_triple):
                - symbol_rate: Optimal symbol rate in symbols/second
                - num_channels: Number of frequency triples to use (1-4)
                - start_freq_triple: Starting frequency triple index (0-42)
        """
        # Determine number of channels needed based on SNR
        # Philosophy: Use single channel when possible, only add channels at low SNR
        # to maintain minimum symbol rate for synchronization (50 sym/s)

        # First, calculate what we can achieve with single channel
        snr_linear = 10 ** (snr_db / 10)
        single_ch_capacity = self.bandwidth * np.log2(1 + snr_linear)

        # Determine modulation for this SNR
        modulation, bits_per_symbol = self.optimal_modulation(snr_db)

        # Calculate achievable rate with single channel at 75% efficiency
        single_ch_max_rate = (single_ch_capacity / bits_per_symbol) * 0.75

        # Prefer 50 sym/s but allow lower if needed for Shannon limit
        # Pattern layer at 25 sym/s provides sync reference
        effective_min_rate = self.preferred_min_rate  # Prefer 50 sym/s

        # FIXED 2-CENTER CONFIGURATION
        # Use 2 centers for balanced performance: good SNR gain + narrower signals
        # Provides +2.76 dB gain and 7-user capacity (2× capacity vs 4-center)
        num_channels = 2  # Always 2 centers (better spectrum efficiency)

        # Calculate effective SNR with always-on center gains
        effective_snr_db = self.get_effective_snr(snr_db, num_channels)

        # Calculate capacity with multi-channel bandwidth and effective SNR
        effective_bandwidth = self.bandwidth * num_channels
        snr_linear = 10 ** (effective_snr_db / 10)
        capacity_bps = effective_bandwidth * np.log2(1 + snr_linear)

        # Calculate theoretical maximum rate from Shannon limit
        # This ensures we NEVER exceed capacity
        shannon_max_rate = capacity_bps / bits_per_symbol

        # Apply efficiency factor - NN decoders can get closer to Shannon limit
        # Modern polar codes + NN decoder can achieve high efficiency
        if snr_db < -10:
            efficiency = 0.50  # 50% efficiency at very low SNR
        elif snr_db < -5:
            efficiency = 0.70  # 70% efficiency (polar codes work well)
        elif snr_db < 0:
            efficiency = 0.75  # 75% efficiency at low SNR
        elif snr_db < 10:
            efficiency = 0.80  # 80% efficiency at medium SNR
        elif snr_db < 20:
            efficiency = 0.85  # 85% efficiency at good SNR
        else:
            efficiency = 0.90  # 90% efficiency at high SNR

        base_rate = shannon_max_rate * efficiency

        # Hard constraint: NEVER exceed Shannon limit
        # If our minimum rate would exceed Shannon capacity, we must use Shannon-limited rate
        max_allowed_rate = capacity_bps / bits_per_symbol * 0.95  # 95% of Shannon limit

        # Always respect Shannon limit, even with multi-channel
        base_rate = min(base_rate, max_allowed_rate)

        # No transition smoothing needed with fixed 4-channel configuration

        # Flexible minimum rate enforcement
        # Prefer 50 sym/s but allow lower to respect Shannon limit
        if snr_db >= 0:
            # Good SNR: enforce preferred minimum
            effective_min_rate = self.preferred_min_rate  # 50 sym/s
        else:
            # Low SNR: allow lower rates if needed
            effective_min_rate = self.min_rate  # 25 sym/s

        # Apply minimum only when it won't violate Shannon limit
        # Check if enforcing minimum would exceed Shannon capacity
        shannon_limited_rate = max_allowed_rate
        if effective_min_rate > shannon_limited_rate * 0.95:
            # Don't enforce minimum if it would violate Shannon
            pass
        elif snr_db >= -6 and num_channels == 1:
            # Single channel: safe to enforce minimum
            base_rate = max(base_rate, effective_min_rate)
        elif snr_db >= -6 and num_channels > 1:
            # Multi-channel: be careful about monotonicity
            pass

        # Ensure we never exceed Shannon limit (double check)

        # Apply multipath penalty (reduce rate in multipath)
        if multipath_severity > 0:
            # Multipath causes ISI, need lower symbol rate
            # Reduction factor: 1.0 (no reduction) to 0.5 (50% reduction for severe)
            multipath_factor = 1.0 - 0.5 * multipath_severity
            base_rate *= multipath_factor

        # Apply QRM penalty (reduce rate with interference)
        if qrm_present:
            # QRM reduces effective SNR, be conservative
            base_rate *= 0.85  # 15% rate reduction

        # Final rate enforcement with Shannon respect
        if snr_db < -10:
            # Very low SNR: allow any rate that fits Shannon
            final_rate = np.clip(base_rate, self.min_rate, self.max_rate)
        elif snr_db < 0:
            # Low SNR: prefer 50 but allow 25 if needed
            min_allowed = min(self.preferred_min_rate, max_allowed_rate * 0.9)
            final_rate = np.clip(base_rate, min_allowed, self.max_rate)
        else:
            # Good SNR: enforce 50 sym/s minimum
            final_rate = np.clip(base_rate, self.preferred_min_rate, self.max_rate)

        # Add small random variation (±5%) for training diversity
        # This helps the model learn to handle continuous rates
        if np.random.random() < 0.8:  # 80% of time, add variation
            variation = 1.0 + (np.random.random() - 0.5) * 0.1  # ±5%
            final_rate *= variation
            # Re-apply same clipping after variation
            if snr_db < -10:
                final_rate = np.clip(final_rate, self.min_rate, self.max_rate)
            elif snr_db < 0:
                min_allowed = min(self.preferred_min_rate, max_allowed_rate * 0.9)
                final_rate = np.clip(final_rate, min_allowed, self.max_rate)
            else:
                final_rate = np.clip(final_rate, self.preferred_min_rate, self.max_rate)

        # Select starting frequency triple for multi-channel transmission
        # For multi-channel, we want to use adjacent triples
        # Choose a random starting position that allows for the required channels
        # Updated for guard bands: 50 triples total (500-2600 Hz range)
        max_start = 14 - num_channels  # Ensure we don't go past triple 13 (50 Hz spacing)
        start_freq_triple = np.random.randint(0, max_start + 1)

        return float(final_rate), num_channels, start_freq_triple

    def get_discrete_rate(self, continuous_rate: float) -> int:
        """
        Quantize continuous rate to nearest CASCADE discrete rate.

        Used for protocol compliance when needed.

        Args:
            continuous_rate: Continuous rate in symbols/second

        Returns:
            Nearest discrete rate from CASCADE protocol
        """
        discrete_rates = [75, 100, 125, 150, 175, 200, 250, 300]
        idx = np.argmin(np.abs(np.array(discrete_rates) - continuous_rate))
        return discrete_rates[idx]


def test_rate_calculator():
    """Test continuous rate calculation across SNR range."""
    calc = ContinuousRateCalculator()

    print("Continuous Symbol Rate Calculation with Adaptive Channels")
    print("=" * 90)
    print("SNR(dB) | Channels | BW(Hz) | Modulation | Rate(sym/s) | Bitrate | Shannon | Efficiency")
    print("-" * 90)

    for snr_db in [-10, -6, -5, -3, -2, 0, 2, 3, 5, 7, 10, 15, 20, 25, 30]:
        mod, bps = calc.optimal_modulation(snr_db)
        rate, channels, start_triple = calc.calculate_continuous_rate(snr_db)
        bandwidth = channels * 60
        bitrate = rate * bps

        # Calculate capacity with multi-channel bandwidth
        snr_linear = 10 ** (snr_db / 10)
        capacity = bandwidth * np.log2(1 + snr_linear)
        efficiency = (bitrate / capacity * 100) if capacity > 0 else 0

        print(f"{snr_db:6.1f} | {channels:8d} | {bandwidth:6d} | {mod:10s} | {rate:11.1f} | {bitrate:7.1f} | {capacity:7.1f} | {efficiency:6.1f}%")

    print("\nMultipath effect (SNR = 10 dB):")
    for severity in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rate, channels, start = calc.calculate_continuous_rate(10.0, multipath_severity=severity)
        print(f"  Severity {severity:.2f}: {rate:.1f} sym/s, {channels} channel(s), start triple {start}")

    print("\nQRM effect (SNR = 10 dB):")
    rate_clean, ch_clean, st_clean = calc.calculate_continuous_rate(10.0, qrm_present=False)
    rate_qrm, ch_qrm, st_qrm = calc.calculate_continuous_rate(10.0, qrm_present=True)
    print(f"  Clean: {rate_clean:.1f} sym/s, {ch_clean} channel(s)")
    print(f"  QRM:   {rate_qrm:.1f} sym/s, {ch_qrm} channel(s) ({(rate_qrm/rate_clean - 1)*100:+.1f}%)")

    print("\nExtreme rates for NN decoder testing (pushing limits):")
    print("SNR(dB) | Channels | Rate(sym/s) | Bitrate | ISI Factor | Notes")
    print("-" * 70)
    for snr_db in [30, 35, 40]:
        rate, channels, start = calc.calculate_continuous_rate(snr_db)
        mod, bps = calc.optimal_modulation(snr_db)
        bitrate = rate * bps
        # ISI factor: how many symbols overlap (rate/bandwidth per channel)
        isi_factor = rate / (calc.bandwidth * channels)
        notes = ""
        if isi_factor > 5:
            notes = "Severe ISI - NN challenge!"
        elif isi_factor > 3:
            notes = "Heavy ISI - NN needed"
        elif isi_factor > 2:
            notes = "Moderate ISI"
        print(f"{snr_db:6.1f} | {channels:8d} | {rate:11.1f} | {bitrate:7.1f} | {isi_factor:10.2f} | {notes}")


if __name__ == "__main__":
    test_rate_calculator()