# Protocol/Model Interface

This directory defines the clean boundary between the protocol layer (discrete decisions) and the model layer (continuous optimization).

## Design Principles

1. **Clear Separation**: Protocol never touches neural networks, model never makes discrete decisions
2. **Type Safety**: Well-defined data structures at boundaries
3. **Streaming**: Natural fragment flow from model to protocol
4. **Stateless**: Each call is independent, no hidden state

## Protocol → Model Interface

### Constraint Passing

The protocol provides constraints that bound the model's optimization:

```python
class ModelConstraints:
    assigned_pattern_pool: str      # Which pool: 'typical_dx', 'good_prop', etc.
    assigned_patterns: List[int]    # Specific patterns from pool (8 patterns)
    priority: float                  # 0.0 (LOW) to 1.0 (EMERGENCY)
    max_time_seconds: float         # Maximum transmission duration
    target_callsign: str            # Destination for link adaptation
    target_kernel: bytes            # 64-bit kernel (includes available_tones)
    multipath_estimate: float       # Estimated delay spread (ms)
```

### Example Usage

```python
# Protocol determines constraints
constraints = ModelConstraints(
    assigned_pattern_pool='typical_dx',  # Most HF operation
    assigned_patterns=[88, 92, 96, 100, 104, 108, 112, 116],  # 8 from typical DX pool
    priority=0.5,  # NORMAL
    max_time_seconds=5.0,
    target_callsign='W2DEF',
    target_kernel=0xABCDEF123456,  # Includes available_tones encoding
    multipath_estimate=5.0  # ms (typical multi-hop)
)

# Pass to model for optimization
encoding_params = model.optimize_encoding(message_data, constraints)
```

## Model → Protocol Interface

### Encoding Parameters

The model returns optimized parameters within protocol constraints:

```python
class EncodingParams:
    selected_patterns: List[int]    # Subset of assigned patterns (from pool)
    pattern_pool: str              # Pool used: 'emergency', 'typical_dx', 'good_prop', 'nvis'
    fragment_duration: float        # Seconds per fragment
    redundancy_factor: float        # FEC strength (1.0-3.0)
    iq_complexity_level: int       # Baked-in complexity of selected patterns (0-7)
    kernel_id: int                 # Natural frame identifier
    available_tones: List[int]     # Which of 70 tones receiver can decode
```

### Streaming Fragments

The model streams fragments to the protocol:

```python
class Fragment:
    data: bytes                    # Encoded fragment data
    duration_seconds: float        # Time to transmit
    patterns_used: List[int]       # Which patterns in this fragment
    kernel_hint_included: bool     # Whether hint is in this fragment

def fragment_generator(frame_data, constraints) -> Iterator[Fragment]:
    """Yields fragments for transmission"""
    processor = model.create_processor(constraints)
    for fragment in processor.process(frame_data):
        yield fragment
```

## Bidirectional Kernel Hints

### Receiver → Transmitter Flow

```python
# 1. Receiver generates hint after decode attempt
def generate_kernel_hint(received_signal, decode_result) -> int:
    """Generate hint to help future transmissions"""
    if decode_result.success_rate < 0.8:
        # Generate hint for challenging conditions
        return model.kernel_generator(received_signal)
    return None

# 2. Protocol includes hint in ACK
ack = {
    'kernel_generated': kernel_hint,
    'for_callsign': 'W1ABC'
}

# 3. Transmitter receives and stores
kernel_cache[ack['for_callsign']] = ack['kernel_generated']

# 4. Transmitter uses hint in constraints
constraints.kernel_hint = kernel_cache.get(target_callsign)
```

## ACK Window Detection

### Model Signals Protocol

```python
class TransmissionPlan:
    fragments: List[Fragment]
    ack_windows: List[AckWindow]

class AckWindow:
    start_time: float              # Seconds from transmission start
    duration: float                # Window duration in seconds
    priority: float                # Importance of ACK here

# Model provides transmission plan with ACK opportunities
plan = model.plan_transmission(data, constraints)
for fragment in plan.fragments:
    transmit(fragment)
    # Check if ACK window follows
    if plan.has_ack_window_after(fragment):
        ack = wait_for_ack(plan.ack_window_duration)
        if ack:
            model.process_ack(ack)
```

## SNR Measurement Interface

### Protocol Measures, Model Adapts

```python
# Protocol performs coarse SNR measurement
def measure_snr(received_signal) -> int:
    """Returns 4-bit coarse SNR bucket"""
    snr_db = calculate_snr(received_signal)
    if snr_db > 10:
        return 3  # +10 dB bucket
    elif snr_db > 0:
        return 2  # 0 dB bucket
    elif snr_db > -10:
        return 1  # -10 dB bucket
    else:
        return 0  # -20 dB bucket

# Model uses measurement for adaptation
def update_link_quality(callsign: str, snr_bucket: int):
    """Update link model with coarse SNR"""
    snr_ranges = [(-30, -10), (-10, 0), (0, 10), (10, 20)]
    snr_min, snr_max = snr_ranges[snr_bucket]
    model.link_estimator.update(callsign, snr_min, snr_max)
```

## Hash Exchange Interface

### Protocol Decides When, Model Unaware

```python
# Protocol determines hash exchange based on SNR
def should_exchange_hashes(link_snr: float, messages_sent: int) -> bool:
    """SNR-scaled hash exchange decision"""
    if link_snr > 10:
        return messages_sent % 10 == 0
    elif link_snr > 0:
        return messages_sent % 25 == 0
    elif link_snr > -10:
        return messages_sent % 50 == 0
    else:
        return False  # Too weak, don't waste bandwidth

# Model only sees result as bandwidth constraint
if should_exchange_hashes(snr, count):
    # Reduce available bandwidth for payload
    constraints.max_time_seconds -= hash_exchange_time
```

## Error Handling

### Protocol Fallbacks

```python
try:
    encoding = model.optimize_encoding(data, constraints)
except ModelError:
    # Fall back to conservative defaults
    encoding = EncodingParams(
        selected_patterns=constraints.assigned_patterns[:4],
        fragment_duration=0.5,
        redundancy_factor=3.0,
        collapse_level=3,  # Binary
        kernel_id=random.randint(0, 2**64-1)
    )
```

### Model Diagnostics

```python
class ModelDiagnostics:
    inference_time_ms: float
    confidence: float
    expert_weights: Dict[str, float]
    warnings: List[str]

# Model provides diagnostics for protocol logging
diagnostics = model.get_diagnostics()
if diagnostics.inference_time_ms > 10:
    log.warning(f"Slow inference: {diagnostics.inference_time_ms}ms")
```

## Testing Interface

### Mock Model for Protocol Testing

```python
class MockModel:
    """Deterministic model for protocol testing"""

    def optimize_encoding(self, data, constraints):
        # Return predictable encoding
        return EncodingParams(
            selected_patterns=constraints.assigned_patterns[:2],
            fragment_duration=1.0,
            redundancy_factor=2.0,
            collapse_level=1,
            kernel_id=0x1234567890ABCDEF
        )
```

### Mock Protocol for Model Testing

```python
class MockProtocol:
    """Simple protocol for model testing"""

    def get_constraints(self):
        return ModelConstraints(
            assigned_patterns=list(range(8)),
            priority=0.5,
            max_time_seconds=5.0,
            target_callsign='TEST',
            kernel_hint=None
        )
```

## Performance Requirements

### Latency Budget
- Constraint creation: <1ms
- Model inference: <10ms
- Fragment generation: Real-time
- ACK processing: <5ms

### Memory Budget
- Constraint object: <1KB
- Fragment buffer: <100KB
- Kernel cache: <10KB
- Total interface: <200KB

This interface ensures clean separation while enabling optimal performance through learned parameters.

## See Also

- **[Protocol Layer](../protocol/README.md)** - Discrete decision-making and protocol constraints
- **[Model Layer](../model/README.md)** - Continuous optimization within constraints
- **[Augmented Inference](augmented_inference.md)** - Runtime inference optimizations
- **[Expert Networks](../model/experts.md)** - How model optimizations are computed