# Smoothness Training Objectives

Training the model to ensure smooth transitions across SNR ranges, preventing jarring changes that could lose synchronization.

Related: [Pattern Complexity Expert](../model/experts.md#pattern-complexity-expert-network) implements these transitions at inference.

## Overview

One of the most challenging aspects of adaptive communication systems is handling transitions. When a system changes modes - whether shifting constellation size, coding rate, or modulation type - there's a risk of losing synchronization. Traditional systems handle this through explicit mode negotiation protocols, adding overhead and complexity.

CASCADE takes a different approach: it trains the model to make naturally smooth transitions that receivers can track without explicit signaling. This is like teaching a driver to gradually change speed rather than slamming on brakes - the passengers (receivers) can adapt to smooth changes but get thrown off by sudden ones.

The smoothness training objectives ensure that as channel conditions change, the model's decisions evolve gradually. This means:
- Pattern complexity changes slowly over time, not in sudden jumps
- The α(SNR) function that controls constellation collapse is smooth and differentiable
- Natural hysteresis prevents oscillation at transition boundaries
- Receivers can track changes through correlation rather than explicit signaling

## Core Principle

The Pattern Complexity Expert must learn smooth α(SNR) functions that gradually transition between constellation levels without discontinuities. This isn't just about mathematical smoothness - it's about ensuring operational continuity in real-world conditions where channels fade and recover.

## Smoothness Loss Components

The smoothness objective consists of multiple complementary loss terms, each addressing a different aspect of smooth operation. These losses work together to create a model that adapts gracefully to changing conditions while maintaining reliable communication.

### 1. Temporal Smoothness

Temporal smoothness ensures that constellation complexity doesn't change too quickly over time. Real channels change gradually - fading happens over seconds or minutes, not microseconds. The model must learn to match this natural tempo.

This loss component penalizes rapid changes between consecutive frames. If the model switches from complex NVIS patterns (IDs 112-127) to emergency patterns (IDs 48-63) instantly, the temporal smoothness loss becomes large. This teaches the model to make gradual transitions between pattern pools (NVIS → Good Prop → Typical DX → Emergency) over several seconds as propagation degrades.

Penalize rapid changes in collapse level:

```python
def temporal_smoothness_loss(alpha_sequence, time_weights):
    """Penalize rapid transitions between frames"""

    # Compute differences between consecutive frames
    alpha_diff = torch.diff(alpha_sequence, dim=0)

    # Weight by time between frames (closer = higher penalty)
    weighted_diff = alpha_diff * time_weights

    # L2 penalty on changes
    return torch.mean(weighted_diff ** 2)
```

### 2. SNR Smoothness

The α(SNR) function that maps signal quality to pattern complexity must be smooth and predictable. This ensures that stations experiencing similar SNR will use similar complexity levels, and that small changes in SNR don't cause large changes in operation.

Without this smoothness constraint, the model might learn a "step function" that suddenly switches modes at specific SNR thresholds. This would cause problems:
- Stations hovering near the threshold would constantly switch modes
- Small measurement errors could cause inappropriate mode changes
- Different stations might disagree on the appropriate mode

By enforcing smoothness in the α(SNR) function, we ensure it looks more like a sigmoid - gradual transitions with stable regions. This creates natural "comfort zones" where the model confidently uses one complexity level.

Ensure gradual transitions across SNR values:

```python
def snr_smoothness_loss(model, snr_range):
    """Ensure smooth α(SNR) function"""

    # Sample SNR values
    snr_values = torch.linspace(snr_range[0], snr_range[1], steps=100)

    # Get collapse decisions
    alpha_values = []
    for snr in snr_values:
        alpha = model.pattern_complexity_expert(snr)
        alpha_values.append(alpha)

    alpha_curve = torch.stack(alpha_values)

    # Penalize second derivative (curvature)
    first_deriv = torch.diff(alpha_curve)
    second_deriv = torch.diff(first_deriv)

    return torch.mean(second_deriv ** 2)
```

### 3. Hysteresis Training

Hysteresis is the property where a system's output depends not just on current input but also on history. For CASCADE, this means the model should be "sticky" - reluctant to change modes unless there's a significant reason. This prevents the maddening oscillation that can occur when conditions hover near a transition point.

Consider a link with SNR fluctuating around 0 dB. Without hysteresis, the model might switch between 16 and 4 patterns every few seconds as SNR crosses the threshold. With hysteresis, the model might need SNR to drop to -2 dB before switching to 4 patterns, but then require it to rise to +2 dB before switching back to 16. This creates a stable 4 dB band where no switching occurs.

Training for hysteresis is subtle - we can't just add a fixed threshold because optimal hysteresis varies with conditions. Instead, we train the model to naturally develop hysteresis by penalizing frequent mode changes within short time windows.

Prevent oscillation at transition boundaries:

```python
class HysteresisObjective:
    """Train model to have natural hysteresis"""

    def __init__(self, hysteresis_width=2.0):
        self.hysteresis_width = hysteresis_width

    def compute_loss(self, model, snr_trajectory):
        """Penalize mode switching within hysteresis band"""

        losses = []
        current_mode = None

        for snr in snr_trajectory:
            predicted_mode = model.get_collapse_level(snr)

            if current_mode is not None:
                # Check if within hysteresis band
                snr_change = abs(snr - self.last_snr)
                if snr_change < self.hysteresis_width:
                    # Penalize mode change
                    if predicted_mode != current_mode:
                        losses.append(1.0)
                    else:
                        losses.append(0.0)

            current_mode = predicted_mode
            self.last_snr = snr

        return torch.mean(torch.tensor(losses))
```

## Training Data Generation

Effective smoothness training requires carefully crafted training data that represents real-world channel behavior. Random noise isn't sufficient - we need realistic patterns of fading, interference, and recovery that match what the model will encounter in deployment.

The training data generation process combines multiple sources:
- Physical channel models (Rayleigh fading, Doppler spread)
- Historical propagation data from WSPR and FT8 logs
- Recorded channel measurements from operational networks
- Synthetic scenarios for edge cases

This diverse training data ensures the model learns robust smoothness properties that work across different propagation conditions, from stable groundwave paths to volatile ionospheric channels.

### Realistic SNR Trajectories

Real channels don't jump randomly between SNR values - they follow physical patterns based on propagation physics. Training data must capture these patterns to teach the model appropriate response dynamics.

Generate training sequences that mimic real channel variations:

```python
def generate_snr_trajectory(duration=300, sample_rate=10):
    """Create realistic SNR variation over time"""

    # Base SNR with slow drift
    time = np.arange(0, duration, 1/sample_rate)
    base_snr = 5 * np.sin(2 * np.pi * time / 300)  # 5-minute cycle

    # Add fading
    rayleigh = np.random.rayleigh(1.0, len(time))
    fading = 10 * np.log10(rayleigh)

    # Add short-term variations
    noise = np.random.normal(0, 1, len(time))

    # Combine
    snr_trajectory = base_snr + fading + noise

    # Smooth to prevent unrealistic jumps
    from scipy.ndimage import gaussian_filter1d
    snr_trajectory = gaussian_filter1d(snr_trajectory, sigma=2)

    return snr_trajectory
```

### Transition Scenarios

While general training covers typical operations, special attention must be paid to transition regions where smoothness is most critical. These boundary conditions are where traditional systems often fail - the model needs extra training to handle them gracefully.

Transition scenarios are deliberately challenging:
- SNR hovering right at a mode boundary for extended periods
- Rapid fading that crosses multiple boundaries quickly
- Slow drift that gradually moves through all complexity levels
- Periodic fading that repeatedly crosses the same boundary

By oversampling these challenging scenarios in training, we ensure the model develops robust strategies for handling transitions rather than just optimizing for stable conditions.

Focus training on boundary conditions:

```python
def create_transition_batch(snr_boundaries=[-10, -5, 0, 5, 10]):
    """Generate training data focused on transitions"""

    batches = []

    for boundary in snr_boundaries:
        # Ascending through boundary
        ascending = np.linspace(boundary - 3, boundary + 3, 60)

        # Descending through boundary
        descending = np.linspace(boundary + 3, boundary - 3, 60)

        # Hovering near boundary
        hovering = boundary + np.random.normal(0, 0.5, 60)

        batches.extend([ascending, descending, hovering])

    return batches
```

## Multi-Objective Training

Training for smoothness involves balancing competing objectives. We want smooth transitions, but not at the expense of communication efficiency. The model must learn when smoothness helps (preventing sync loss) and when it hurts (staying in suboptimal mode too long).

The multi-objective loss function carefully weights different goals:
- **Primary objective**: Minimize bit error rate (communication works)
- **Smoothness constraints**: Gentle enough to track but responsive to changes
- **Efficiency bounds**: Don't sacrifice too much throughput for smoothness

Finding the right balance is crucial. Too much emphasis on smoothness creates a sluggish system that can't adapt to rapidly changing conditions. Too little creates a jumpy system that loses synchronization during transitions.

### Combined Loss Function

The complete training objective combines multiple loss components with carefully tuned weights:

Balance efficiency with smoothness:

```python
def cascade_training_loss(output, target, alpha_history):
    """Complete training objective"""

    # Primary: Communication efficiency
    efficiency_loss = compute_ber(output, target)

    # Smoothness constraints
    temporal_smooth = temporal_smoothness_loss(alpha_history)
    snr_smooth = snr_smoothness_loss(model, [-25, 15])
    hysteresis = hysteresis_objective.compute_loss(model, snr_trajectory)

    # Weighted combination
    total_loss = (
        1.0 * efficiency_loss +      # Primary objective
        0.1 * temporal_smooth +       # Smooth over time
        0.05 * snr_smooth +          # Smooth α(SNR) function
        0.05 * hysteresis            # Prevent oscillation
    )

    return total_loss
```

### Curriculum Learning

Curriculum learning - training on progressively harder tasks - is particularly effective for smoothness objectives. We can't expect the model to learn smooth transitions before it learns basic communication. The training must progress through stages, each building on the previous.

This staged approach mirrors how humans learn complex skills:
1. First, learn to communicate at all (basic functionality)
2. Then, learn to maintain communication (stability)
3. Finally, learn to transition smoothly (elegance)

By gradually introducing smoothness constraints, we ensure the model has a solid foundation before tackling the subtle challenge of smooth adaptation.

Gradually introduce smoothness constraints:

```python
class SmoothnessCurriculum:
    """Gradually increase smoothness requirements"""

    def __init__(self, total_epochs=100):
        self.total_epochs = total_epochs

    def get_smoothness_weight(self, epoch):
        """Ramp up smoothness penalty"""

        if epoch < 20:
            # Phase 1: Learn basic functionality
            return 0.0
        elif epoch < 50:
            # Phase 2: Introduce smoothness
            return 0.05 * (epoch - 20) / 30
        else:
            # Phase 3: Full smoothness
            return 0.05 + 0.05 * (epoch - 50) / 50
```

## Validation Metrics

Validating smoothness training requires specialized metrics that go beyond simple accuracy measurements. We need to quantify how gracefully the model handles transitions and whether receivers can maintain synchronization through mode changes.

These validation metrics are run on held-out test data that includes challenging real-world scenarios not seen during training. This ensures the model has learned generalizable smoothness properties rather than memorizing specific trajectories.

### Smoothness Metrics

Quantifying smoothness requires multiple complementary measurements:

Measure quality of transitions:

```python
def evaluate_smoothness(model, test_trajectories):
    """Quantify transition smoothness"""

    metrics = {}

    for trajectory in test_trajectories:
        alpha_sequence = []
        for snr in trajectory:
            alpha = model.pattern_complexity_expert(snr)
            alpha_sequence.append(alpha)

        # Measure oscillations
        changes = np.diff(alpha_sequence)
        reversals = np.sum(np.diff(np.sign(changes)) != 0)
        metrics['oscillations'] = reversals / len(trajectory)

        # Measure abruptness
        max_change = np.max(np.abs(changes))
        metrics['max_change'] = max_change

        # Measure hysteresis width
        # (SNR range where mode is stable)
        metrics['hysteresis_width'] = measure_hysteresis(model)

    return metrics
```

### Synchronization Testing

The ultimate test of smoothness is whether receivers maintain synchronization through transitions. This goes beyond mathematical smoothness to operational success. A transition might look smooth in metrics but still cause sync loss if it violates receiver assumptions.

Synchronization testing uses a realistic receiver model that matches deployed hardware capabilities. We simulate the complete communication chain including:
- Channel estimation and tracking loops
- Pattern correlation and detection
- Adaptive equalization
- Symbol timing recovery

By testing against realistic receivers, we ensure the model's transitions work in practice, not just in theory.

Verify receiver can track smooth transitions:

```python
def test_synchronization(model, channel_sim):
    """Test if receiver maintains sync through transitions"""

    # Generate challenging trajectory
    snr_trajectory = generate_transition_test()

    sync_maintained = True
    for i, snr in enumerate(snr_trajectory):
        # Transmit with current collapse level
        tx_data = generate_test_data()
        tx_signal = model.encode(tx_data, snr)

        # Simulate channel
        rx_signal = channel_sim(tx_signal, snr)

        # Attempt decode
        rx_data, sync_status = model.decode(rx_signal)

        if not sync_status:
            sync_maintained = False
            break

    return sync_maintained, i / len(snr_trajectory)
```

## Training Schedule

The training schedule carefully orchestrates the learning process to achieve both communication efficiency and smooth adaptation. This isn't just about epoch counts - it's about strategically introducing complexity as the model develops capability.

### Three-Phase Approach

The three-phase training schedule reflects the natural learning progression:

#### Phase 1 (Epochs 1-30): Performance Focus
**Objective**: Establish core communication capability

During this phase, the model learns the fundamentals:
- How to encode and decode using CASCADE patterns
- Basic channel compensation and noise suppression
- Pattern selection for different SNR levels

No smoothness constraints are applied yet. The model is free to make abrupt transitions if they improve performance. This ensures we don't handicap the model before it learns to communicate effectively.

**Success Criteria**: Achieve 75% Shannon efficiency (realistic for async multi-user CDMA/FHSS)

#### Phase 2 (Epochs 31-70): Smoothness Introduction
**Objective**: Learn graceful adaptation while maintaining performance

This phase gradually introduces smoothness requirements:
- Smoothness weight starts at 0 and increases linearly
- Model learns to balance efficiency with stability
- Discovers natural transition strategies

The gradual introduction is critical. If we suddenly apply full smoothness constraints, the model's performance would collapse. The slow ramp allows the model to find smooth solutions that still communicate effectively.

**Success Criteria**: Smoothness metrics improve while maintaining >85% Shannon efficiency

#### Phase 3 (Epochs 71-100): Fine-Tuning
**Objective**: Polish transitions and optimize for edge cases

The final phase focuses on perfection:
- Full smoothness weights applied
- Extra training on challenging transition scenarios
- Fine-tune hysteresis behavior
- Optimize for worst-case conditions

This phase uses specialized training data that emphasizes boundary conditions and challenging scenarios. The model learns to handle edge cases that might be rare but critical for robust operation.

**Success Criteria**: <0.1% sync loss during transitions, 2-4 dB hysteresis bands

## Expected Outcomes

After successful smoothness training, the model exhibits sophisticated adaptive behavior that balances multiple objectives. These learned behaviors emerge naturally from the training process rather than being explicitly programmed.

### Learned Behaviors

The trained model discovers several key strategies:

1. **Natural Hysteresis**: ~3 dB bands where mode is stable
   - Model learns different hysteresis widths for different SNR regions
   - Wider bands in volatile conditions, narrower in stable
   - Prevents oscillation without explicit thresholds

2. **Predictive Transitions**: Change mode before quality degrades
   - Model learns to anticipate channel degradation from subtle cues
   - Begins transitions early to maintain communication
   - Uses recent history to predict near-future conditions

3. **Smooth α(SNR) Curves**: Sigmoid-like transition functions
   - Natural S-curves emerge from training
   - Stable plateaus at each complexity level
   - Gradual transitions between plateaus

4. **Temporal Stability**: Resist rapid mode changes
   - Model becomes naturally "sticky" in its decisions
   - Requires sustained evidence before changing modes
   - Filters out short-term channel variations

### Performance Targets

Concrete metrics that indicate successful smoothness training:

- **Mode Changes**: <1 per minute under typical fading
  - Real channels rarely require frequent mode switches
  - Model learns to ride out temporary variations

- **Hysteresis Width**: 2-4 dB depending on SNR region
  - Wider at mode boundaries (e.g., 4 dB around 0 dB SNR)
  - Narrower in stable regions (e.g., 2 dB around +10 dB SNR)

- **Sync Loss Rate**: <0.1% during transitions
  - Receivers successfully track 99.9% of mode changes
  - Critical for maintaining reliable communication

- **Efficiency Penalty**: <2% due to smoothness constraints
  - Small throughput sacrifice for large stability gain
  - System remains near-optimal while being smooth

## Implementation Notes

Implementing smoothness training presents unique challenges compared to standard supervised learning. The temporal nature of smoothness requires processing sequences rather than individual samples, which impacts memory usage and training efficiency.

### GPU Memory Optimization

Smoothness training requires processing long trajectory sequences to evaluate temporal consistency. This creates memory challenges - a 5-minute trajectory at 10 Hz sampling is 3000 time steps. Processing multiple trajectories simultaneously for batch training can quickly exhaust GPU memory.

The solution is gradient accumulation - process smaller sub-batches and accumulate gradients before updating weights. This maintains the statistical benefits of large batch training while fitting within memory constraints:

Smoothness training requires trajectory batches:

```python
def efficient_trajectory_training(model, trajectories, batch_size=8):
    """Memory-efficient trajectory training"""

    # Use gradient accumulation for long sequences
    accumulation_steps = 4

    optimizer.zero_grad()
    for i, trajectory_batch in enumerate(trajectories):
        # Process sub-batch
        loss = compute_trajectory_loss(model, trajectory_batch)
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

### Online Smoothness Adaptation

While the model is trained with fixed smoothness objectives, real-world deployment benefits from adaptive smoothness based on current conditions. A stable ground-wave path needs less smoothness than a volatile ionospheric circuit. The model can adjust its smoothness behavior based on observed channel stability.

This online adaptation doesn't require retraining - it's a simple parameter adjustment that scales the learned smoothness behavior. Think of it like adjusting the suspension stiffness in a car based on road conditions:

Allow runtime smoothness adjustment:

```python
class AdaptiveSmoothness:
    """Adjust smoothness based on conditions"""

    def __init__(self, model):
        self.model = model
        self.stability_history = []

    def update_smoothness(self, channel_stability):
        """Adapt based on channel conditions"""

        if channel_stability < 0.3:
            # Unstable channel: More smoothness
            self.model.smoothness_factor = 1.5
        elif channel_stability > 0.7:
            # Stable channel: Allow faster transitions
            self.model.smoothness_factor = 0.5
        else:
            # Normal smoothness
            self.model.smoothness_factor = 1.0
```

## Benefits

The smoothness training objectives provide crucial operational advantages:

1. **Robust Operation**: Maintains sync through fading
   - Receivers can track gradual transitions without losing lock
   - Communication continues through changing conditions
   - Reduces need for reacquisition protocols

2. **Predictable Behavior**: Operators understand mode changes
   - Smooth transitions are intuitive to human operators
   - Easy to predict system behavior from channel conditions
   - Simplified troubleshooting and debugging

3. **Efficient Transitions**: Minimal overhead during changes
   - No need for explicit mode negotiation protocols
   - Transitions happen within normal data flow
   - Saves bandwidth compared to step-change systems

4. **Natural Adaptation**: Responds smoothly to conditions
   - Matches the gradual nature of real propagation changes
   - Avoids overreacting to temporary disturbances
   - Maintains optimal performance across varying conditions

5. **Hardware Friendly**: Reduces abrupt power changes
   - Smooth transitions are easier on RF amplifiers
   - Reduces stress on power supplies
   - Minimizes splatter and spurious emissions

## Design Philosophy

Smoothness training embodies CASCADE's philosophy of learning natural behaviors rather than imposing rigid rules. Traditional systems use fixed thresholds and explicit state machines for mode switching. CASCADE learns when and how to transition through experience with real channels.

This learned smoothness is more robust than designed smoothness because it adapts to the actual statistics of channel behavior rather than theoretical models. The model discovers transition strategies that work in practice, including subtle behaviors we might not think to program explicitly.

## Practical Impact

In operational deployment, smoothness training makes the difference between a system that constantly drops and reacquires sync versus one that maintains continuous communication through challenging conditions. Users experience this as the difference between choppy, interrupted communication and smooth, reliable connectivity.

For emergency communications where every second counts, the ability to maintain sync through transitions can be literally life-saving. The small efficiency penalty paid for smoothness is vastly outweighed by the reliability gained.

## See Also

- **[Pattern Complexity Expert](../model/experts.md#pattern-complexity-expert-network)** - Network that implements smooth transitions
- **[Hysteresis Prevention](../model/experts.md#hysteresis-prevention)** - Preventing mode oscillation
- **[Training README](README.md)** - Overall training strategy including smoothness objectives