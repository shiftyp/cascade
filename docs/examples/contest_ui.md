# CASCADE Contest User Interface

This document describes the recommended user interface for CASCADE contest and high-activity operations, emphasizing station quality ranking and real-time transmission previews.

## Overview

CASCADE's UI differs from traditional amateur radio software by providing **predictive performance metrics** for each station based on measured SNR, exchanged kernels, and hardware capabilities. This enables operators to optimize their contest strategy by targeting the most efficient contacts.

## Main Interface Components

### 1. Connected Stations List (Real-Time)

**Live display of all detected stations with predicted performance:**

```
═══════════════════════════════════════════════════════════════════
CASCADE Contest Mode - 20m Band (14.074 MHz)          My HW: RPi+Coral
═══════════════════════════════════════════════════════════════════

Connected Stations (47 visible) - Sorted by Throughput ▼

┌────┬──────────┬──────┬──────────┬─────────┬──────┬─────────────┐
│ St │ Callsign │ SNR  │ Est. BPS │ Msg Time│ QSO× │ Status      │
├────┼──────────┼──────┼──────────┼─────────┼──────┼─────────────┤
│[●]│ W2DEF    │ +12  │  8,400   │  1.6s   │  12× │ 8QAM/Coral  │ ← Click
│[●]│ K5XYZ    │ +10  │  7,200   │  1.6s   │   8× │ 8QAM/Desk   │
│[●]│ N7ABC    │ +8   │  5,100   │  3.2s   │   6× │ QPSK/Coral  │
│[◐]│ W1MNO    │ +5   │  3,200   │  3.2s   │   5× │ QPSK/RPi    │
│[◐]│ VK2ZOI   │ +3   │  1,800   │  4.8s   │   3× │ BPSK/Coral  │
│[○]│ JA1XYZ   │ -2   │    600   │  6.4s   │   2× │ BPSK/RPi    │
│[○]│ ZS6ABC   │ -8   │    200   │ 16.0s   │   1× │ BPSK/Unk    │
│[~]│ VK4MNO   │ -15  │    150   │ 24.0s   │   1× │ FT8-mode    │
│   │ ...      │      │          │         │      │             │
└────┴──────────┴──────┴──────────┴─────────┴──────┴─────────────┘

Legend:
[●] Excellent (8-QAM capable, >+8 dB)
[◐] Good (QPSK capable, 0-8 dB)
[○] Fair (BPSK only, -10-0 dB)
[~] Weak (FT8-mode, <-10 dB)

Sort by: [Throughput▼] [QSO Multi] [SNR] [Call] [Grid]
Filter: [Show kernel-mode only] [Show all] [DX only (>5000km)]

Auto-select top 15 for rapid-fire? [Start] [Configure]
═══════════════════════════════════════════════════════════════════
```

**Column explanations:**
- **St**: Status indicator (connection quality)
- **Callsign**: Station identifier
- **SNR**: Measured signal-to-noise ratio
- **Est. BPS**: Predicted throughput to this station (from model)
- **Msg Time**: Estimated time for typical message (96 bytes)
- **QSO×**: Multiplier effect (how many other stations ACK when you work this one)
- **Status**: Modulation/Hardware tier inferred from kernel

### 2. Message Composition with Live Preview

**Real-time transmission estimate as operator types:**

```
═══════════════════════════════════════════════════════════════════
Compose Message

To: W2DEF [●] +12 dB (8-QAM capable)                    [Select ▼]

Message:
┌─────────────────────────────────────────────────────────────────┐
│ W2DEF DE K0BB RST 599 NAME BOB QTH COLORADO K                  │ ← Typing here
│                                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Transmission Preview:
├─ Estimated time: 3.2 seconds (2 patterns)
├─ Modulation: 8-QAM (based on W2DEF's kernel)
├─ Patterns used: 5, 12 (assigned to you)
├─ Data: 312 bits (39 bytes) + FEC → 768 bits
├─ Target SNR: +12 dB (excellent)
└─ Confidence: High (kernel 2 minutes old)

Expected ACKs from: W2DEF, K5XYZ, N7ABC (3× QSO multiplier)

[Send] [Cancel] [Save Draft]
═══════════════════════════════════════════════════════════════════
```

**Updates in real-time:**
- Add text → "3.8 seconds (3 patterns)"
- Delete text → "1.6 seconds (1 pattern)"
- Select different target → Recalculates based on their kernel
- Shows confidence level (kernel age, SNR stability)

### 3. Activity Monitor (Received Messages)

**Real-time message stream with decode metrics:**

```
═══════════════════════════════════════════════════════════════════
Activity Monitor - Last 60 seconds

15:23:45 [●] W2DEF → K5XYZ: "RST 599" (8-QAM, 1.6s, +12 dB)
15:23:42 [◐] N7ABC → ALL: "QRZ?" (QPSK, 1.6s, +5 dB)
15:23:40 [●] K0BB → W2DEF: "599 BOB CO" (8-QAM, 3.2s, +10 dB) ← You
15:23:35 [○] VK2ZOI → ZL1: "TU 73" (BPSK, 4.8s, -3 dB)
15:23:30 [~] Beacon: JA1XYZ (4-FSK, 1.3s, -18 dB)
15:23:28 [●] K5XYZ → W2DEF: "5NN TX" (8-QAM, 1.6s, +11 dB)

Users decoded this minute: 23 / 47 detected (hardware limit: 50)
Beacons heard: 12
Network throughput (personal): 8,450 bps aggregate

[Filter: All] [Messages Only] [Beacons Only] [My QSOs]
═══════════════════════════════════════════════════════════════════
```

### 4. Contest Strategy Dashboard

**Optimize for maximum QSOs/hour:**

```
═══════════════════════════════════════════════════════════════════
Contest Strategy Optimizer

Current Rate: 145 QSOs/hour (last 10 minutes)
Target: 200 QSOs/hour

Strategy Recommendations:
┌────────────────────────────────────────────────────────────────┐
│ ✓ Target high-multiplier stations (12× avg)                   │
│   → W2DEF, K5XYZ, N7ABC recommended                           │
│                                                                │
│ ⚠ Avoid low-multiplier stations (<3×)                         │
│   → VK2ZOI, JA1XYZ only if needed for mult                    │
│                                                                │
│ ⚠ 8 stations in FT8-mode (>20s per QSO)                       │
│   → Skip unless rare DX multiplier                            │
│                                                                │
│ ✓ 23 stations in kernel-mode (<5s per QSO)                    │
│   → Focus here for maximum rate                               │
└────────────────────────────────────────────────────────────────┘

Predicted improvement: +40 QSOs/hour if focusing on kernel-mode only

[Apply Auto-Select] [Manual] [Show Details]
═══════════════════════════════════════════════════════════════════
```

### 5. Network Topology View

**Visual representation of connected stations:**

```
═══════════════════════════════════════════════════════════════════
Network Topology (My Station: K0BB, FN42)

            Strong Links (>+5 dB, kernel-mode)
                   ╱  W2DEF (+12)
                  ╱   K5XYZ (+10)
                 ╱    N7ABC (+8)
          K0BB ─┤     W1MNO (+5)
                 ╲
                  ╲   Fair Links (0-5 dB, kernel-mode)
                   ╲  VE3ABC (+3)
                    ╲ W6XYZ (+2)

            Weak Links (<0 dB, FT8-mode only)
                      VK2ZOI (-3)
                      JA1XYZ (-8)
                      ZS6MNO (-15)

Relay Opportunities:
- W2DEF can relay to: 15 stations I don't hear
- I can relay for: 8 stations (W2DEF, K5XYZ to VK/JA)

[Show Full Mesh] [Relay Mode] [Geographic View]
═══════════════════════════════════════════════════════════════════
```

## Key UI Features

### Predictive Performance

**Model-based throughput prediction:**
```python
def calculate_station_metrics(station):
    """Predict performance for UI display"""

    kernel = kernel_cache.get(station.callsign)

    # Model predicts link performance
    prediction = model.predict_link_quality(
        target_kernel=kernel,
        measured_snr=station.snr,
        my_hardware=my_tier,
        their_hardware=infer_from_kernel(kernel)
    )

    return {
        'bps': prediction.throughput,          # 200-11,000 bps
        'msg_time': prediction.message_latency, # 1.6-24s
        'modulation': prediction.modulation,    # 8-QAM/QPSK/BPSK
        'qso_multiplier': prediction.ack_count, # 1-15×
        'success_rate': prediction.reliability  # 0.7-0.99
    }
```

### Auto-Target Selection

**Contest mode auto-selects optimal stations:**
```python
def auto_select_contest_targets(available_stations, goal='max_qsos_per_hour'):
    """Select stations to maximize contest score"""

    if goal == 'max_qsos_per_hour':
        # Prioritize high multiplier × fast QSO
        ranked = sorted(stations,
                       key=lambda s: s.qso_multiplier / s.msg_time,
                       reverse=True)

    elif goal == 'max_throughput':
        # Prioritize raw BPS
        ranked = sorted(stations, key=lambda s: s.bps, reverse=True)

    elif goal == 'rare_multipliers':
        # Prioritize new grids/states
        ranked = sorted(stations, key=lambda s: s.mult_value, reverse=True)

    return ranked[:15]  # Top 15 for rapid-fire
```

### Live Transmission Visualization

**Show signal as it transmits:**
```
Transmitting to W2DEF...

Pattern 5: ████████░░░░░░░░ (1.6s / 3.2s)
Pattern 12: ░░░░░░░░░░░░░░░░ (waiting)

Frequency: [███ message tones ░ beacon gaps ███]
Modulation: 8-QAM → QPSK (adapting to interference from K5XYZ)

ACKs received: 2 / ~8 expected
├─ W2DEF (target): ✓ Confirmed (+12 dB)
├─ K5XYZ: ✓ Also heard (+9 dB)
└─ Waiting for others...
```

## Implementation Notes

**UI runs model inference for predictions:**
- Prediction is fast (<5ms, no actual encoding)
- Updates as kernels refresh
- Shows confidence based on kernel age
- Real-time adaptation to changing conditions

**Storage requirements:**
- Beacon cache: ~50 stations × 100 bytes = 5KB
- Kernel cache: ~50 stations × 32 bytes = 1.6KB
- Activity log: Last 1000 messages × 200 bytes = 200KB
- Total: <250KB (trivial)

See [hardware_requirements.md](../deployment/hardware_requirements.md) for deployment specifications and [signal_specification.md](../protocol/signal_specification.md) for protocol details.
