"""
CASCADE Shannon Limit Comparison - CONTINUOUS ADAPTIVE SYMBOL RATE

Symbol rate continuously optimized in 1 sym/s increments to achieve
~90% Shannon efficiency across all SNR levels with smooth transitions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class AdaptiveCascadeSystem:
    """CASCADE system with continuously adaptive symbol rate, modulation, and coding"""
    
    def __init__(self):
        self.pattern_rate = 75  # sym/s (fixed)
        self.channel_spacing = 20  # Hz
        self.max_bandwidth = 300  # Hz (max total bandwidth to use)
        self.min_symbol_rate = 10  # sym/s (minimum practical)
        self.max_symbol_rate = 300  # sym/s (maximum practical)
        
        # Available modulations and FEC rates
        self.modulations = [
            ("BPSK", 1),
            ("QPSK", 2),
            ("8PSK", 3),
            ("16APSK", 4),
        ]
        
        self.fec_rates = [1/3, 1/2, 2/3, 3/4, 5/6]
    
    def combining_gain(self, num_carriers: int, orthogonal: bool = False, 
                      diversity: bool = False, snr_db: float = 0) -> float:
        """SNR improvement from combining multiple carriers"""
        if num_carriers == 1:
            return 0.0
        
        ideal_gain_db = 10 * np.log10(num_carriers)
        
        if orthogonal:
            base_efficiency = 0.70
            snr_penalty = 0.90 if snr_db < -15 else 1.0
            actual_efficiency = base_efficiency * snr_penalty
            return ideal_gain_db * actual_efficiency
        elif diversity:
            correlation_factor = 0.6
            actual_gain = ideal_gain_db * correlation_factor
            diversity_gain_db = 1.5
            return actual_gain + diversity_gain_db
        else:
            base_efficiency = 0.30
            if snr_db < -15:
                snr_penalty = 0.8
            elif snr_db < -10:
                snr_penalty = 0.9
            else:
                snr_penalty = 1.0
            actual_efficiency = base_efficiency * snr_penalty
            return ideal_gain_db * actual_efficiency
    
    def shannon_limit(self, bandwidth_hz: float, snr_db: float) -> float:
        """Shannon capacity in bps"""
        snr_linear = 10 ** (snr_db / 10)
        return bandwidth_hz * np.log2(1 + snr_linear)
    
    def signal_bandwidth(self, symbol_rate: float, rolloff: float = 0.1) -> float:
        """Required bandwidth for given symbol rate"""
        return (1 + rolloff) * symbol_rate
    
    def total_bandwidth(self, num_carriers: int, signal_bw: float, orthogonal: bool = False) -> float:
        """Total occupied bandwidth"""
        if num_carriers == 1:
            return signal_bw
        
        if orthogonal:
            # N carriers occupy (2N-1) slots of 20 Hz each
            num_slots = 2 * num_carriers - 1
            return num_slots * 20
        else:
            # Non-orthogonal: 40 Hz spacing
            return signal_bw + (num_carriers - 1) * 40
    
    def optimize_configuration(self, snr_db: float, orthogonal: bool = False, 
                              max_carriers: int = 8, target_efficiency: float = 0.90) -> dict:
        """
        Optimize symbol rate, modulation, FEC, and number of carriers.
        Target ~90% efficiency but prioritize smooth throughput scaling.
        """
        best_config = None
        best_score = -float('inf')
        
        # Try different numbers of carriers
        for num_carriers in range(1, max_carriers + 1):
            # Effective SNR after combining
            gain_db = self.combining_gain(num_carriers, orthogonal, False, snr_db)
            eff_snr_db = snr_db + gain_db
            
            # Try different modulations and FEC rates
            for mod_name, bits_per_sym in self.modulations:
                for fec_rate in self.fec_rates:
                    # Find optimal symbol rate for this configuration
                    config = self._optimize_symbol_rate(
                        num_carriers, eff_snr_db, snr_db,
                        mod_name, bits_per_sym, fec_rate,
                        orthogonal, target_efficiency
                    )
                    
                    if config:
                        # Scoring: balance throughput and efficiency target
                        throughput = config['coded_bps']
                        efficiency = config['efficiency']
                        efficiency_error = abs(efficiency - target_efficiency)
                        
                        # Score favors high throughput AND being close to target efficiency
                        # Use multiplicative penalty for efficiency deviation
                        efficiency_penalty = 1.0 / (1.0 + 5.0 * efficiency_error)
                        score = throughput * efficiency_penalty
                        
                        if score > best_score:
                            best_score = score
                            best_config = config
        
        return best_config if best_config else self._fallback_config(snr_db, orthogonal)
    
    def _optimize_symbol_rate(self, num_carriers: int, eff_snr_db: float, snr_db: float,
                              mod_name: str, bits_per_sym: int, fec_rate: float,
                              orthogonal: bool, target_efficiency: float) -> Optional[dict]:
        """
        Find optimal symbol rate for given configuration.
        Symbol rate in integer increments of 1 sym/s.
        Find rate closest to target efficiency.
        """
        # Quick check: can this modulation work at this SNR?
        required_snr_db = (bits_per_sym - 1) * 4 - 10 * np.log10(fec_rate) - 8
        if eff_snr_db < required_snr_db:
            return None
        
        best_rate = None
        best_config = None
        best_efficiency_error = float('inf')
        
        # Try all integer symbol rates from min to max
        for symbol_rate in range(self.min_symbol_rate, self.max_symbol_rate + 1):
            # Calculate bandwidths
            sig_bw = self.signal_bandwidth(symbol_rate)
            total_bw = self.total_bandwidth(num_carriers, sig_bw, orthogonal)
            
            # Check bandwidth constraint
            if total_bw > self.max_bandwidth:
                break  # No point trying higher rates
            
            # Calculate throughput and Shannon limit
            coded_bps = symbol_rate * bits_per_sym * fec_rate
            shannon_bps = self.shannon_limit(total_bw, eff_snr_db)
            
            if shannon_bps <= 0.01:
                continue
            
            efficiency = coded_bps / shannon_bps
            
            # Must be physically possible (can't exceed Shannon limit)
            if efficiency > 0.98:
                continue
            
            # Track configuration closest to target efficiency
            # This ensures smooth efficiency across SNR range
            efficiency_error = abs(efficiency - target_efficiency)
            if efficiency_error < best_efficiency_error:
                best_efficiency_error = efficiency_error
                best_rate = symbol_rate
                best_config = {
                    'sig_bw': sig_bw,
                    'total_bw': total_bw,
                    'coded_bps': coded_bps,
                    'shannon_bps': shannon_bps,
                    'efficiency': efficiency
                }
        
        if best_config is None or best_rate is None:
            return None
        
        # Build final configuration
        gain_db = self.combining_gain(num_carriers, orthogonal, False, snr_db)
        num_slots = (2 * num_carriers - 1) if orthogonal and num_carriers > 1 else num_carriers
        
        return {
            'num_carriers': num_carriers,
            'num_slots': num_slots,
            'snr_db': snr_db,
            'combining_gain_db': gain_db,
            'effective_snr_db': eff_snr_db,
            'modulation': mod_name,
            'bits_per_symbol': bits_per_sym,
            'fec_rate': fec_rate,
            'symbol_rate': best_rate,
            'signal_bandwidth_hz': best_config['sig_bw'],
            'total_bandwidth_hz': best_config['total_bw'],
            'coded_bps': best_config['coded_bps'],
            'shannon_bps': best_config['shannon_bps'],
            'efficiency': best_config['efficiency'],
            'orthogonal': orthogonal
        }
    
    def _fallback_config(self, snr_db: float, orthogonal: bool) -> dict:
        """Fallback to most robust configuration"""
        num_carriers = 8
        gain_db = self.combining_gain(num_carriers, orthogonal, False, snr_db)
        eff_snr_db = snr_db + gain_db
        symbol_rate = 25
        sig_bw = self.signal_bandwidth(symbol_rate)
        total_bw = self.total_bandwidth(num_carriers, sig_bw, orthogonal)
        coded_bps = symbol_rate * 1 * (1/3)
        shannon_bps = self.shannon_limit(total_bw, eff_snr_db)
        
        num_slots = (2 * num_carriers - 1) if orthogonal else num_carriers
        
        return {
            'num_carriers': num_carriers,
            'num_slots': num_slots,
            'snr_db': snr_db,
            'combining_gain_db': gain_db,
            'effective_snr_db': eff_snr_db,
            'modulation': 'BPSK',
            'bits_per_symbol': 1,
            'fec_rate': 1/3,
            'symbol_rate': symbol_rate,
            'signal_bandwidth_hz': sig_bw,
            'total_bandwidth_hz': total_bw,
            'coded_bps': coded_bps,
            'shannon_bps': shannon_bps,
            'efficiency': coded_bps / shannon_bps if shannon_bps > 0 else 0,
            'orthogonal': orthogonal
        }


def comparison_orthogonal_vs_nonorthogonal():
    """Compare orthogonal vs non-orthogonal modes"""
    system = AdaptiveCascadeSystem()
    
    print("\n" + "="*180)
    print("ORTHOGONAL vs NON-ORTHOGONAL COMPARISON - CONTINUOUS ADAPTIVE")
    print("="*180)
    print("\nSymbol rate continuously optimized in 1 sym/s increments to target 90% Shannon efficiency")
    print("Maximum bandwidth: 300 Hz\n")
    print("Non-Orthogonal: 2-FSK pairs, 40 Hz spacing, 30% combining efficiency")
    print("Orthogonal: Same data on all carriers, (2N-1) slots × 20 Hz, 70% combining efficiency\n")
    
    snr_values = [-20, -15, -10, -5, 0, +5, +10]
    
    print(f"{'SNR':<6} {'Mode':<15} {'Carriers':<9} {'Slots':<7} {'Gain':<7} {'Eff SNR':<9} {'Sym/s':<8} {'Mod':<8} "
          f"{'FEC':<6} {'Tot BW':<9} {'Coded':<10} {'Shannon':<10} {'Eff %':<8}")
    print("-" * 180)
    
    for snr in snr_values:
        # Non-orthogonal
        result_non = system.optimize_configuration(snr, orthogonal=False, max_carriers=8)
        print(f"{snr:<6.0f} {'Non-orthogonal':<15} {result_non['num_carriers']:<9} "
              f"{'-':<7} {result_non['combining_gain_db']:<7.1f} {result_non['effective_snr_db']:<9.1f} "
              f"{result_non['symbol_rate']:<8} {result_non['modulation']:<8} "
              f"{result_non['fec_rate']:<6.2f} {result_non['total_bandwidth_hz']:<9.1f} "
              f"{result_non['coded_bps']:<10.1f} {result_non['shannon_bps']:<10.1f} "
              f"{result_non['efficiency']*100:<8.1f}%")
        
        # Orthogonal
        result_orth = system.optimize_configuration(snr, orthogonal=True, max_carriers=8)
        print(f"{snr:<6.0f} {'Orthogonal':<15} {result_orth['num_carriers']:<9} "
              f"{result_orth['num_slots']:<7} {result_orth['combining_gain_db']:<7.1f} "
              f"{result_orth['effective_snr_db']:<9.1f} {result_orth['symbol_rate']:<8} "
              f"{result_orth['modulation']:<8} {result_orth['fec_rate']:<6.2f} "
              f"{result_orth['total_bandwidth_hz']:<9.1f} {result_orth['coded_bps']:<10.1f} "
              f"{result_orth['shannon_bps']:<10.1f} {result_orth['efficiency']*100:<8.1f}%")
        
        # Show improvement
        throughput_improvement = result_orth['coded_bps'] / result_non['coded_bps']
        bw_savings = result_non['total_bandwidth_hz'] - result_orth['total_bandwidth_hz']
        print(f"{'':6} {'→ Benefit':<15} {'':<9} {'':<7} {'':<7} {'':<9} {'':<8} {'':<8} {'':<6} "
              f"{bw_savings:<9.1f}Hz  {throughput_improvement:<10.2f}× rate")
        print()


def detailed_shannon_comparison():
    """Detailed comparison across SNR range"""
    system = AdaptiveCascadeSystem()
    
    print("\n" + "="*180)
    print("CASCADE CONTINUOUS ADAPTIVE ANALYSIS - ORTHOGONAL MODE")
    print("="*180)
    print("\nSymbol rate optimized in 1 sym/s increments to target 90% Shannon efficiency")
    print("Maximum bandwidth: 300 Hz\n")
    
    snr_values = np.arange(-20, 11, 1)
    
    print(f"{'SNR':<6} {'Carriers':<9} {'Slots':<7} {'Gain':<7} {'Eff SNR':<9} {'Sym/s':<8} {'Mod':<8} "
          f"{'FEC':<6} {'Tot BW':<9} {'Coded':<10} {'Shannon':<10} {'Efficiency':<12}")
    print("-" * 180)
    
    results = []
    for snr in snr_values:
        result = system.optimize_configuration(
            snr_db=float(snr), 
            orthogonal=True,
            max_carriers=8
        )
        
        print(f"{result['snr_db']:<6.0f} {result['num_carriers']:<9} {result['num_slots']:<7} "
              f"{result['combining_gain_db']:<7.1f} {result['effective_snr_db']:<9.1f} "
              f"{result['symbol_rate']:<8} {result['modulation']:<8} {result['fec_rate']:<6.2f} "
              f"{result['total_bandwidth_hz']:<9.1f} {result['coded_bps']:<10.1f} "
              f"{result['shannon_bps']:<10.1f} {result['efficiency']*100:<12.1f}%")
        
        results.append(result)
    
    # Check for any >100% efficiency
    over_100 = [r for r in results if r['efficiency'] > 1.0]
    if over_100:
        print("\n⚠️  WARNING: Found efficiency > 100% at these SNR values:")
        for r in over_100:
            print(f"   SNR {r['snr_db']:.0f} dB: {r['efficiency']*100:.1f}%")
    else:
        print("\n✓ All efficiencies ≤ 100% (valid)")
    
    # Check efficiency consistency
    efficiencies = [r['efficiency'] for r in results]
    avg_eff = np.mean(efficiencies)
    std_eff = np.std(efficiencies)
    print(f"\nEfficiency statistics:")
    print(f"  Mean: {avg_eff*100:.1f}%")
    print(f"  Std dev: {std_eff*100:.1f}%")
    print(f"  Range: {min(efficiencies)*100:.1f}% - {max(efficiencies)*100:.1f}%")
    
    return results


def validate_calculation():
    """Validate the continuous adaptive optimization"""
    system = AdaptiveCascadeSystem()
    
    print("\n" + "="*100)
    print("VALIDATION: Bandwidth calculation")
    print("="*100)
    
    print("\nOrthogonal mode - slot calculation:")
    print("-" * 50)
    for num_carriers in [1, 2, 3, 4, 8]:
        num_slots = 2 * num_carriers - 1 if num_carriers > 1 else 1
        bw_hz = num_slots * 20
        print(f"  {num_carriers} carriers → {num_slots} slots × 20 Hz = {bw_hz:.0f} Hz")
    
    print("\n" + "="*100)
    print("VALIDATION: Continuous Adaptive Optimization (Target: 90% efficiency)")
    print("="*100)
    
    test_snrs = [-20, -15, -14, -13, -12, -10, -5, 0, +5, +10]
    
    for snr in test_snrs:
        print(f"\n{'='*100}")
        print(f"SNR = {snr} dB")
        print(f"{'='*100}")
        
        for orth, mode_name in [(False, "Non-Orthogonal"), (True, "Orthogonal")]:
            print(f"\n{mode_name} Mode:")
            print("-" * 50)
            
            result = system.optimize_configuration(snr, orthogonal=orth, max_carriers=8)
            
            print(f"  Carriers: {result['num_carriers']}")
            if orth:
                print(f"  Slots: {result['num_slots']} (2×{result['num_carriers']}-1)")
            print(f"  Combining gain: {result['combining_gain_db']:.2f} dB")
            print(f"  Effective SNR: {result['effective_snr_db']:.2f} dB")
            print(f"  Optimized symbol rate: {result['symbol_rate']} sym/s")
            print(f"  Modulation: {result['modulation']}")
            print(f"  FEC rate: {result['fec_rate']:.3f}")
            print(f"  Signal BW: {result['signal_bandwidth_hz']:.1f} Hz")
            print(f"  Total BW: {result['total_bandwidth_hz']:.1f} Hz")
            print(f"  Coded throughput: {result['coded_bps']:.2f} bps")
            print(f"  Shannon limit: {result['shannon_bps']:.2f} bps")
            print(f"  Efficiency: {result['efficiency']*100:.2f}% (target: 90%)")


def compare_to_ft8():
    """Compare CASCADE to FT8"""
    system = AdaptiveCascadeSystem()
    
    print("\n" + "="*150)
    print("CASCADE vs FT8 COMPARISON - CONTINUOUS ADAPTIVE (90% Shannon efficiency)")
    print("="*150)
    
    ft8_throughput = 6.1  # bps
    msg_bits = 77
    
    print(f"\nFT8: {ft8_throughput} bps at -21 dB\n")
    
    snr_values = [-20, -15, -10, -5, 0, +5, +10]
    
    for mode_name, orth in [("Non-Orthogonal", False), ("Orthogonal", True)]:
        print(f"\n{mode_name} Mode:")
        print(f"{'SNR (dB)':<10} {'Carriers':<10} {'Slots':<8} {'Sym/s':<10} {'Mod':<10} "
              f"{'Throughput':<15} {'vs FT8':<12} {'Msg Time':<12}")
        print("-" * 150)
        
        for snr in snr_values:
            result = system.optimize_configuration(snr, orthogonal=orth, max_carriers=8)
            speedup = result['coded_bps'] / ft8_throughput
            msg_time = msg_bits / result['coded_bps']
            slots_str = str(result['num_slots']) if orth else '-'
            
            print(f"{snr:<10} {result['num_carriers']:<10} {slots_str:<8} "
                  f"{result['symbol_rate']:<10} {result['modulation']:<10} "
                  f"{result['coded_bps']:<15.2f} {speedup:<12.2f}× {msg_time:<12.2f}s")


def plot_shannon_efficiency():
    """Plot efficiency analysis"""
    system = AdaptiveCascadeSystem()
    
    snr_range = np.arange(-25, 15, 0.5)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Collect data for both modes
    data = {}
    for mode_name, orth in [("Non-Orthogonal", False), ("Orthogonal", True)]:
        coded_rates = []
        shannon_limits = []
        carriers_used = []
        efficiencies = []
        symbol_rates = []
        
        for snr in snr_range:
            result = system.optimize_configuration(snr, orthogonal=orth, max_carriers=8)
            coded_rates.append(result['coded_bps'])
            shannon_limits.append(result['shannon_bps'])
            carriers_used.append(result['num_carriers'])
            efficiencies.append(result['efficiency'] * 100)
            symbol_rates.append(result['symbol_rate'])
        
        data[mode_name] = {
            'coded': coded_rates,
            'shannon': shannon_limits,
            'carriers': carriers_used,
            'efficiency': efficiencies,
            'symbol_rate': symbol_rates
        }
    
    # Plot 1: Throughput comparison
    ax1 = axes[0, 0]
    ax1.plot(snr_range, data['Non-Orthogonal']['coded'], '-', linewidth=2, 
            label='Non-Orth Achieved', color='blue')
    ax1.plot(snr_range, data['Orthogonal']['coded'], '-', linewidth=2, 
            label='Orthogonal Achieved', color='green')
    ax1.plot(snr_range, data['Non-Orthogonal']['shannon'], '--', linewidth=1, 
            alpha=0.6, color='blue', label='Non-Orth Shannon')
    ax1.plot(snr_range, data['Orthogonal']['shannon'], '--', linewidth=1, 
            alpha=0.6, color='green', label='Orth Shannon')
    ax1.set_xlabel('SNR (dB)', fontsize=11)
    ax1.set_ylabel('Data Rate (bps)', fontsize=11)
    ax1.set_title('Throughput: Continuous Adaptive (90% Target)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-25, 15)
    ax1.axvline(-20, color='red', linestyle=':', alpha=0.5)
    
    # Plot 2: Symbol rate adaptation
    ax2 = axes[0, 1]
    ax2.plot(snr_range, data['Non-Orthogonal']['symbol_rate'], linewidth=2, 
            label='Non-Orthogonal', color='blue')
    ax2.plot(snr_range, data['Orthogonal']['symbol_rate'], linewidth=2, 
            label='Orthogonal', color='green')
    ax2.set_xlabel('SNR (dB)', fontsize=11)
    ax2.set_ylabel('Symbol Rate (sym/s)', fontsize=11)
    ax2.set_title('Adaptive Symbol Rate (1 sym/s increments)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-25, 15)
    ax2.axvline(-20, color='red', linestyle=':', alpha=0.5)
    
    # Plot 3: Efficiency
    ax3 = axes[1, 0]
    ax3.plot(snr_range, data['Non-Orthogonal']['efficiency'], linewidth=2, 
            label='Non-Orthogonal', color='blue')
    ax3.plot(snr_range, data['Orthogonal']['efficiency'], linewidth=2, 
            label='Orthogonal', color='green')
    ax3.axhline(100, color='red', linestyle='--', alpha=0.5, linewidth=1.5, 
               label='Shannon limit (100%)')
    ax3.axhline(90, color='orange', linestyle=':', alpha=0.5, linewidth=1.5, 
               label='Target (90%)')
    ax3.set_xlabel('SNR (dB)', fontsize=11)
    ax3.set_ylabel('Shannon Efficiency (%)', fontsize=11)
    ax3.set_title('Coding Efficiency (Target: 90%)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-25, 15)
    ax3.set_ylim(0, 120)
    ax3.axvline(-20, color='red', linestyle=':', alpha=0.5)
    
    # Plot 4: Carrier count
    ax4 = axes[1, 1]
    ax4.plot(snr_range, data['Non-Orthogonal']['carriers'], linewidth=2, 
            label='Non-Orthogonal', color='blue')
    ax4.plot(snr_range, data['Orthogonal']['carriers'], linewidth=2, 
            label='Orthogonal', color='green')
    ax4.set_xlabel('SNR (dB)', fontsize=11)
    ax4.set_ylabel('Number of Carriers', fontsize=11)
    ax4.set_title('Adaptive Carrier Count', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-25, 15)
    ax4.set_ylim(0, 9)
    ax4.axvline(-20, color='red', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plot_file = '/tmp/cascade_continuous_adaptive.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to {plot_file}")
    
    return plot_file


if __name__ == "__main__":
    validate_calculation()
    detailed_shannon_comparison()
    comparison_orthogonal_vs_nonorthogonal()
    compare_to_ft8()
    
    plot_file = plot_shannon_efficiency()
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Plot: {plot_file}")