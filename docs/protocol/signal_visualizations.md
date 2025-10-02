# CASCADE Signal Visualizations

Visual representations of CASCADE's signal structure in IQ, frequency, and time domains.

## IQ Constellation Diagrams

### Base Constellations

**8-QAM (High SNR, Messages):**
```
            Q (Quadrature)
                 ↑
                 |
         Point 7 o           o Point 6
                 |╲         ╱|
                 | ╲       ╱ |
                 |  ╲     ╱  |
                 |   ╲   ╱   |
   Point 5 o─────┼────┼────┼─────o Point 4
         ╱       |   ╱ ╲   |       ╲
        ╱        |  ╱   ╲  |        ╲
       ╱         | ╱     ╲ |         ╲
      ╱          |╱       ╲|          ╲
Point 3 o        o─────────o           o Point 2
                 |   I
                 |  (In-phase)
         Point 1 o
                 |
         Point 0 o

8 constellation points
3 bits per symbol (log₂(8))
Typical spacing: 1.0 units between points
```

**QPSK (Medium SNR):**
```
            Q
            ↑
            |
   Point 3  o           o  Point 2
            |╲         ╱|
            | ╲       ╱ |
            |  ╲     ╱  |
            |   ╲   ╱   |
────────────┼────┼────┼──────────→ I
            |   ╱ ╲   |
            |  ╱   ╲  |
            | ╱     ╲ |
            |╱       ╲|
   Point 1  o           o  Point 0
            |

4 constellation points (corners)
2 bits per symbol
Spacing: √2 units (maximum separation)
```

**BPSK (Low SNR):**
```
            Q
            |
            |
            |
   Point 1  |         Point 0
    o───────┼───────────o────→ I
            |
            |
            |

2 constellation points (I-axis only)
1 bit per symbol
Spacing: 2.0 units (maximum possible)
```

## Continuous Constellation Collapse

### Morphing Animation Sequence

**How 8-QAM continuously becomes BPSK:**

```
t=0s (SNR +15 dB):              t=10s (SNR +5 dB):              t=20s (SNR -5 dB):
8-QAM, full spread              Intermediate (6-QAM-like)       QPSK-like

    Q                               Q                               Q
    ↑                               ↑                               ↑
  7 o   o 6                       7'o o'6                          · · ·
    ·   ·                           ·o·                            o   o
────o─·─o──── I                 ────o─o──── I                  ────o─o──── I
    ·   ·                           ·o·                            o   o
  3 o   o 2                       3'o o'2                          · · ·

Spacing: 1.0                    Spacing: 0.6                    Spacing: 0.3
8 distinct points               6-7 distinguishable             4 distinguishable
3 bits/symbol                   2-2.5 bits/symbol               2 bits/symbol


t=30s (SNR -15 dB):             t=40s (SNR -20 dB):
BPSK-like                       Pure BPSK

    Q                               Q
    ↑                               ↑
    ····                            |
    o─o                             |
────o─o──── I                   o───┼───o── I
    o─o                             |
    ····                            |

Spacing: 0.1                    Spacing: 0.05 (collapsed)
2 distinguishable               2 on I-axis only
1 bit/symbol                    1 bit/symbol
```

**Physical process**: Constellation points continuously move inward (toward origin) as SNR degrades. Closer spacing = fewer reliably distinguishable points = lower information rate.

## Frequency Domain - Tone Allocation

### Channel Allocation Overview

**Complete 2.5 kHz CASCADE channel assignment (300-2800 Hz optimal passband):**

```
                    CASCADE Channel Allocation
                    ══════════════════════════

300 Hz ══════════════════════════════════════════════════════ 2800 Hz
│                                                                   │
├─ 300 Hz ────── Message 1 (8-QAM, 50ms symbols)
│
├─ 378 Hz ────── Normal Beacon 1 (4-FSK, 160ms symbols)
│
├─ 456 Hz ────── EMERGENCY 1 (4-FSK, 800ms, RESERVED) ◄── Includes
│                                                               grid
├─ 534 Hz ────── Normal Beacon 2 (4-FSK, 160ms symbols)         square
│
├─ 612 Hz ────── Message 2 (8-QAM, 50ms symbols)
│
├─ 768 Hz ────── EMERGENCY 2 (4-FSK, 800ms, RESERVED) ◄──
│
├─ 925 Hz ────── Message 3 (8-QAM, 50ms symbols)
│
├─ 1081 Hz ───── EMERGENCY 3 (4-FSK, 800ms, RESERVED) ◄──
│
├─ 1237 Hz ───── Message 4 (8-QAM, 50ms symbols)
│
├─ 1393 Hz ───── EMERGENCY 4 (4-FSK, 800ms, RESERVED) ◄──
│
├─ 1550 Hz ───── Message 5 (8-QAM, 50ms symbols)
│
├─ 1862 Hz ───── Message 6 (8-QAM, 50ms symbols)
│
├─ 2018 Hz ───── Normal Beacon 3 (4-FSK, 160ms symbols)
│
├─ 2175 Hz ───── Message 7 (8-QAM, 50ms symbols)
│
├─ 2253 Hz ───── Normal Beacon 4 (4-FSK, 160ms symbols)
│
├─ 2487 Hz ───── Message 8 (8-QAM, 50ms symbols)
│
2800 Hz ══════════════════════════════════════════════════════════

Summary:
• 8 Message tones    [300, 612, 925, 1237, 1550, 1862, 2175, 2487] Hz
• 4 Normal beacons   [378, 534, 2018, 2253] Hz
• 4 Emergency (RESERVED) [456, 768, 1081, 1393] Hz
• Total: 16 tones in 2.5 kHz channel
• Guard spacing: 78-156 Hz between different signal types
• NOTE: Shifted +300 Hz from baseband to avoid AC coupling issues in SSB transceivers
```

### 2.5 kHz Spectrum Layout

```
Frequency (Hz)
│
2800├─────────────────────────────────────────────────┤ Channel edge
    │                                                 │
2487├─── █ Message Tone 8                            │
    │                                                 │
2253├─────── ▓ Normal Beacon Tone 4                  │
    │                                                 │
2175├─── █ Message Tone 7                            │
    │                                                 │
2018├─────── ▓ Normal Beacon Tone 3                  │
    │                                                 │
1862├─── █ Message Tone 6                            │
    │                                                 │
1550├─── █ Message Tone 5                            │
    │                                                 │
1393├───────────── ◆ EMERGENCY 4 (RESERVED)          │
    │                                                 │
1237├─── █ Message Tone 4                            │
    │                                                 │
1081├───────────── ◆ EMERGENCY 3 (RESERVED)          │
    │                                                 │
 925├─── █ Message Tone 3                            │
    │                                                 │
 768├───────────── ◆ EMERGENCY 2 (RESERVED)          │
    │                                                 │
 612├─── █ Message Tone 2                            │
    │                                                 │
 534├─────── ▓ Normal Beacon Tone 2                  │
    │                                                 │
 456├───────────── ◆ EMERGENCY 1 (RESERVED)          │
    │                                                 │
 378├─────── ▓ Normal Beacon Tone 1                  │
    │                                                 │
 300├─── █ Message Tone 1                            │
    └─────────────────────────────────────────────────┘

Legend:
█ = Message Tones (8 tones, 64 patterns, 8-QAM adaptive)
▓ = Normal Beacon (4 tones, 4-FSK, LDPC 1/2)
◆ = Emergency (4 tones, RESERVED, 4-FSK, LDPC 1/4, includes grid square)

Bandwidth utilization:
- Messages: ~160 Hz (8 × 20 Hz per tone @ 50ms symbols)
- Normal beacons: ~24 Hz (4 × 6 Hz per tone @ 160ms symbols)
- Emergency: ~5 Hz (4 × 1.25 Hz per tone @ 800ms symbols)
- Total: ~189 Hz / 2500 Hz = 7.6% occupied
- Remaining: Guard bands, filter roll-off
- NOTE: Frequency shift (+300 Hz) avoids AC coupling issues in SSB radios
```

## Time-Frequency Grid

### Single Pattern Transmission

**Message Pattern 12 (50ms symbols, 32 symbols = 1.6s):**

```
Freq (Hz)
2487 ┤ █  ·  █  ·  ·  █  ·  █  ·  ·  █  ·  █  ·  ·  ...  (Symbol sequence)
2175 ┤ ·  █  ·  ·  █  ·  █  ·  ·  █  ·  ·  █  ·  █  ...
1862 ┤ ·  ·  ·  █  ·  ·  ·  ·  █  ·  ·  ·  █  ·  ·  ...
1550 ┤ █  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  █  ·  ·  ·  ...
1237 ┤ ·  ·  ·  ·  ·  ·  ·  █  ·  ·  █  ·  ·  ·  ·  ...
 925 ┤ ·  ·  █  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ...
 612 ┤ ·  ·  ·  ·  ·  ·  █  ·  ·  ·  ·  ·  ·  ·  █  ...
 300 ┤ ·  ·  ·  ·  █  ·  ·  ·  ·  █  ·  ·  ·  █  ·  ...
     └─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──
      0 50 100 150 200 250 300 350 400... (Time in ms)

Each column = one 50ms symbol
Active tone (█) depends on pattern sequence
Pattern 12's unique sequence creates orthogonality with other patterns
```

### Multi-User Overlay (3 users simultaneously)

```
Freq (Hz)
2487 ┤ A  B  AC ·  C  AB ·  A  BC ·  ABC ...  (A, B, C transmitting)
2175 ┤ B  ·  B  A  ·  C  AB ·  A  C  B   ...
1862 ┤ ·  C  ·  BC A  ·  B  ·  ·  A   C  ...
1550 ┤ C  A  ·  B  B  ·  C  BC ·  ·   A  ...
1237 ┤ ·  ·  ·  C  ·  ·  A  ·  A  C   B  ...
 925 ┤ ·  ·  A  ·  ·  B  ·  C  B  ·   ·  ...
 612 ┤ ·  ·  C  ·  ·  ·  C  B  ·  B   ·  ...
 300 ┤ ·  ·  B  ·  A  A  ·  ·  C  ·   ·  ...
     └─┬──┬──┬──┬──┬──┬──┬──┬──┬──┬───────
      0 50 100 150 200 250 300 350 400 (ms)

User A (Pattern 5): Uses tone sequence specific to Pattern 5
User B (Pattern 12): Different tone sequence (orthogonal to A)
User C (Pattern 23): Different sequence (orthogonal to A and B)

Where A, B, C overlap (same tone, same time):
- Signals add in IQ plane
- Model separates via pattern correlation
- Orthogonality (<-30 dB) ensures clean separation
```

## Composite IQ Plane (What Sound Card Sees)

**Received signal = superposition of all active users:**

```
            Q
            ↑
            |        * Composite signal (sum of 50 users)
         *** | ***   Cloud of overlapping constellation points
       **  *|*  **
      *   **┼**   *  Each * represents contribution from one user's symbol
      * ****┼**** *  Model must separate this chaos into individual signals
    **──────┼──────**────→ I
      * ****┼**** *
      *   **┼**   *
       **  *|*  **
         *** | ***
            |

Single user (clean 8-QAM):
    o   o   8 distinct points
    o   o   Easy to decode

50 users (overlapped):
    ****    Cloud of ~400 points (50 users × 8 points each)
   ******   Signal Expert must:
   ******   1. Correlate against 64 patterns
   ***  *   2. Identify which points belong to which user
            3. Decode each user's constellation
            4. Return 50 separate decoded messages

Model accomplishes this via:
- Pattern correlation (primary separation, -30 dB)
- Constellation geometry (secondary, learned from kernels)
- Temporal features (onset timing, envelope)
- Frequency signatures (clock drift unique per radio)
```

## Waterfall View (Time-Frequency)

**5-second CASCADE channel view:**

```
Freq
(Hz)
2800 ─┬─────────────────────────────────────────────────────────────
      │ ░░░░░░░░░░░░░░░░░░░░░░░░░  (Noise floor)
2487 ─┤ ████████████░░██████░░░█████████  Message Tone 8 (multiple users)
      │
2253 ─┤ ░░░▓▓▓░░░░░░░░░░▓▓▓░░░░░░░  Beacon Tone 4 (slower symbols)
      │
2175 ─┤ ████████████░░██████░░░█████████  Message Tone 7
      │
2018 ─┤ ░░░▓▓▓░░░░░░░░░░▓▓▓░░░░░░░  Beacon Tone 3
      │
1862 ─┤ ████████████░░██████░░░█████████  Message Tone 6
      │
1550 ─┤ ████████████░░██████░░░█████████  Message Tone 5
      │
1393 ─┤ ░░░░░░░░░░◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆  Emergency 4 (very long symbols)
      │
1237 ─┤ ████████████░░██████░░░█████████  Message Tone 4
      │
1081 ─┤ ░░░░░░░░░░◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆  Emergency 3
      │
 925 ─┤ ████████████░░██████░░░█████████  Message Tone 3
      │
 768 ─┤ ░░░░░░░░░░◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆  Emergency 2
      │
 612 ─┤ ████████████░░██████░░░█████████  Message Tone 2
      │
 534 ─┤ ░░░▓▓▓░░░░░░░░░░▓▓▓░░░░░░░  Beacon Tone 2
      │
 456 ─┤ ░░░░░░░░░░◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆  Emergency 1
      │
 378 ─┤ ░░░▓▓▓░░░░░░░░░░▓▓▓░░░░░░░  Beacon Tone 1
      │
 300 ─┤ ████████████░░██████░░░█████████  Message Tone 1
     ─┴─────┬────┬────┬────┬────┬─────── Time (seconds)
           0    1    2    3    4    5

█ = Message activity (dense, 50ms symbols)
▓ = Beacon activity (sparse, 160ms symbols)
◆ = Emergency (very sparse, 800ms symbols, continuous)
░ = Noise/idle

Multiple message users create dense █ regions
Beacons create periodic ▓ bursts
Emergency creates very long ◆ bars (44.8s duration, includes grid square)
NOTE: All frequencies shifted +300 Hz to avoid AC coupling in SSB transceivers
```

## Multi-User IQ Overlay

### 3 Users Transmitting Simultaneously

**User A (Pattern 5, high SNR):**
```
Q ↑
  |    Points widely spaced (8-QAM)
7 o    o 6
  ·    ·
──o─·──o─→ I
  ·    ·
3 o    o 2
```

**User B (Pattern 12, medium SNR):**
```
Q ↑
  |    Points moderately spaced
  · o··o ·
  ·  ··  ·
──o──oo──o─→ I
  ·  ··  ·
  · o··o ·
```

**User C (Pattern 23, low SNR):**
```
Q ↑
  |    Collapsed toward BPSK
  |····
  |o··o
──|oooo|─→ I
  |o··o
  |····
```

**Composite (A + B + C received simultaneously):**
```
Q ↑
  |
  *★*o★*
  ★*oo*★
──o★ooo★o─→ I
  ★*oo*★
  *★*o★*

o = User A's points (clear 8-QAM)
· = User B's points (intermediate)
★ = User C's points (collapsed)
* = Overlapping regions (multiple users)

Signal Expert sees this composite
Separates via:
1. Pattern correlation (which tones active when)
2. Constellation geometry (point spacing)
3. Temporal features (symbol timing)
```

## Beacon vs Message vs Emergency (Overlaid)

**Same time, different frequencies:**

```
            IQ Plane View

    Q ↑
      |
  ◆◆◆◆┼◆◆◆◆  Emergency (4-FSK, 4 points, [456,768,1081,1393] Hz)
      |
  ▓▓▓▓┼▓▓▓▓  Beacons (4-FSK, 4 points, [378,534,2018,2253] Hz)
      |
  ████┼████  Messages (8-QAM, 8 points, [300,612,925,...] Hz)
 ██████████
──────┼──────→ I
 ██████████
  ████┼████
      |
  ▓▓▓▓┼▓▓▓▓
      |
  ◆◆◆◆┼◆◆◆◆

Each signal type occupies different IQ region
(because different frequencies = different complex phases)

Model separates by:
- Frequency (bandpass filtering)
- Symbol rate (50ms vs 160ms vs 800ms) - temporal orthogonality
- Bandwidth (20 Hz vs 6 Hz vs 1.25 Hz) - spectral orthogonality
- Modulation depth (8-QAM vs 4-FSK emergency vs 4-FSK beacon)
```

## Pattern Orthogonality Visualization

**How 64 patterns remain orthogonal:**

```
Correlation Matrix (64 × 64):

       Pattern 0  Pattern 1  Pattern 2  ...  Pattern 63
       ─────────  ─────────  ─────────       ──────────
P0  │     1.0       -0.002      0.001    ...   -0.001    │
P1  │   -0.002       1.0       -0.003   ...    0.002    │
P2  │    0.001     -0.003        1.0    ...   -0.001    │
... │     ...        ...         ...            ...      │
P63 │   -0.001      0.002      -0.001   ...     1.0     │

Diagonal: 1.0 (perfect self-correlation)
Off-diagonal: <0.003 (corresponds to <-30 dB cross-correlation)

This orthogonality allows 64 users to transmit simultaneously
Model can separate via correlation peak detection
```

## Constellation Collapse Mechanism

**How model transitions from 8-QAM to BPSK while encoding:**

### Information Rate Adaptation

```python
# Model doesn't just move points - it adapts information rate

# High SNR encoding:
data = 0b11010110  # 8 bits
symbols_8qam = [
    map_to_8qam(0b110),  # First 3 bits → 8-QAM point
    map_to_8qam(0b101),  # Next 3 bits → 8-QAM point
    map_to_8qam(0b10?)   # Last 2 bits + padding
]
# 3 symbols carry 8 bits = 2.67 bits/symbol average

# Low SNR encoding (same 8 bits):
# Constellation collapsed, only 2 points distinguishable
# Model adapts: Use repetition coding
symbols_bpsk = [
    map_to_bpsk(0b1), map_to_bpsk(0b1), map_to_bpsk(0b1),  # Bit 0, repeated 3×
    map_to_bpsk(0b1), map_to_bpsk(0b1), map_to_bpsk(0b1),  # Bit 1, repeated 3×
    map_to_bpsk(0b0), map_to_bpsk(0b0), map_to_bpsk(0b0),  # Bit 2, repeated 3×
    // ... 24 symbols total for 8 bits
]
# 24 symbols carry 8 bits = 0.33 bits/symbol

# Constellation collapse + repetition = graceful degradation
```

**Physical process:**
1. **Constellation geometry collapses** (points move closer)
2. **Fewer points distinguishable** (8 points → 4 → 2)
3. **Encoder adapts bit rate** (uses fewer bits per symbol)
4. **Adds repetition** (same bits sent multiple times)
5. **Symbol rate can stretch** (50ms → 100ms for extra robustness)

**Result**: Continuous adaptation from 3 bits/symbol (fast) to 0.33 bits/symbol (robust)

## See Also

- **[Signal Specification](signal_specification.md)** - Technical parameters
- **[Adaptive 4-FSK](adaptive_4fsk.md)** - Control channel details
- **[Model Architecture](../model/README.md)** - How model processes these signals
