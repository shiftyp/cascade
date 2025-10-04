# Priority Handling Protocol

Priority is determined by the user and can be modified during relay. The protocol makes discrete routing decisions while the model optimizes transmission parameters.

## Overview

Message priority is a fundamental challenge in any communication system. Too rigid and the system can't adapt to emergencies; too flexible and important messages get lost in the noise. CASCADE's approach puts users in control while allowing the network to make intelligent adjustments when needed.

The key innovation is separating priority decisions between protocol and model layers:
- **Protocol** (discrete): Decides queue order, relay permission, and priority upgrades
- **Model** (continuous): Optimizes encoding strength, redundancy, and timing within priority classes

This separation ensures predictable priority handling that users can understand and trust, while still benefiting from learned optimization. It's like having traffic laws (protocol) that determine right of way, while individual drivers (model) optimize their exact speed and lane position.

Priority can be modified during relay - a crucial feature for emergency networks. A routine message containing emergency keywords can be automatically upgraded by relay stations. This creates a self-organizing emergency response system without central control.

## Priority Levels

CASCADE defines four priority levels that cover the full range of amateur radio traffic. These aren't arbitrary categories but reflect decades of emergency communication experience:

User-assigned priority hints guide both protocol and model decisions:

```python
class Priority(Enum):
    EMERGENCY = 0  # Life safety traffic
    HIGH = 1       # Urgent operational
    NORMAL = 2     # Standard messages
    LOW = 3        # Bulk/background
```

## User Assignment

The system trusts users to set appropriate priorities - there's no algorithmic second-guessing of user intent. This respects the amateur radio tradition of operator responsibility while providing guidance for appropriate use.

### Message Creation

When creating a message, users explicitly choose the priority level. The interface should make this choice prominent, not hidden in advanced options. Priority is as important as the destination address:

Users set initial priority:

```python
def create_message(content, destination, priority=Priority.NORMAL):
    """User determines message priority"""

    return {
        'from': my_callsign,
        'to': destination,
        'id': generate_id(),
        'priority': priority,
        'data': content,
        'timestamp': time.time()
    }
```

### Priority Guidelines

While users have final say over priority, the system provides clear guidelines to promote consistent use across the network. These aren't enforced rules but community norms that emerge from operational experience:

Suggested to users, not enforced:

| Priority | Use Cases | Examples |
|----------|-----------|----------|
| EMERGENCY | Life safety | "Medical emergency at...", "Wildfire approaching..." |
| HIGH | Time-critical ops | "Weather warning", "Net control traffic" |
| NORMAL | Standard comms | "QSO traffic", "Regular check-ins" |
| LOW | Non-urgent | "QSL confirmations", "Telemetry data" |

## Relay Priority Modification

One of CASCADE's most important features is allowing relay stations to upgrade message priority based on content or conditions. This creates resilience - even if the originating station doesn't recognize an emergency, the network can adapt. This is particularly valuable when non-radio-savvy users access the network through gateways.

The modification is one-way: priority can only be increased, never decreased. This prevents well-meaning but misguided attempts to "clean up" priority assignments. Once someone declares an emergency, it stays an emergency.

### Protocol Decision

The protocol layer makes discrete decisions about priority modification. These are rule-based, predictable, and auditable. Operators can understand exactly why a priority was changed:

Relay stations can upgrade priority based on content analysis or conditions:

```python
class RelayProtocol:
    """Protocol-level relay decisions"""

    def should_modify_priority(self, message, current_conditions):
        """Relay station may upgrade priority"""

        # Emergency keywords trigger upgrade
        emergency_keywords = ['emergency', 'mayday', 'urgent', 'sos']
        content_lower = message['data'].lower()

        if any(keyword in content_lower for keyword in emergency_keywords):
            if message['priority'] != Priority.EMERGENCY:
                return Priority.EMERGENCY, "Contains emergency keywords"

        # Weather alerts during severe conditions
        if 'weather' in content_lower and current_conditions['severe_weather']:
            if message['priority'] == Priority.NORMAL:
                return Priority.HIGH, "Weather info during alert"

        # Aged messages might get upgraded
        age = time.time() - message['timestamp']
        if age > 3600 and message['priority'] == Priority.LOW:
            return Priority.NORMAL, "Aged message upgrade"

        return None, None  # No modification

    def relay_message(self, message, link_quality):
        """Process message for relay"""

        # Check if priority should be modified
        new_priority, reason = self.should_modify_priority(
            message,
            self.current_conditions
        )

        if new_priority is not None:
            # Log the modification
            self.log_priority_change(message['id'],
                                   message['priority'],
                                   new_priority,
                                   reason)

            # Update priority
            message['priority'] = new_priority
            message['relay_modified'] = True
            message['modification_reason'] = reason

        return message
```

### Downgrade Protection

The protocol enforces a strict rule: priority can only be upgraded, never downgraded. This protection ensures that important messages don't get buried by erroneous relay decisions. It reflects the principle that it's better to over-prioritize than to miss an emergency.

This one-way modification also simplifies the protocol - relay stations don't need to make complex judgments about whether to lower priority. They only need to recognize when elevation is appropriate:

Priority can only be upgraded, never downgraded:

```python
def validate_priority_change(original, proposed):
    """Ensure priority only increases"""

    if proposed.value < original.value:  # Lower value = higher priority
        return proposed
    else:
        return original  # Keep original if not an upgrade
```

## Model Optimization

While the protocol handles discrete priority decisions, the model learns continuous optimization strategies for each priority level. This is where machine learning shines - discovering the optimal balance of redundancy, power, and timing for different priority traffic.

The model doesn't decide what priority a message has (that's protocol), but it learns how to best transmit messages of each priority. High priority messages might get 3× redundancy while low priority gets 0.7×. These multipliers aren't fixed but learned from experience.

### Priority-Weighted Training

During training, the model sees realistic priority distributions and learns that errors on high-priority messages are more costly than errors on low-priority ones:

The model learns to optimize for different priorities:

```python
def priority_weighted_loss(output, target, priority):
    """Weight loss by message priority"""

    base_loss = compute_ber(output, target)

    # Higher weight for higher priority
    priority_weights = {
        Priority.EMERGENCY: 10.0,
        Priority.HIGH: 3.0,
        Priority.NORMAL: 1.0,
        Priority.LOW: 0.3
    }

    return base_loss * priority_weights[priority]
```

### Adaptive Encoding

The model learns sophisticated strategies for encoding different priority levels. This goes beyond simple redundancy increases - the model might discover that emergency messages benefit from different pattern selections or fragment timings:

Model adjusts parameters based on priority:

```python
class PriorityAwareEncoder:
    """Model adapts encoding to priority"""

    def select_redundancy(self, priority, snr):
        """Higher priority gets more redundancy"""

        base_redundancy = self.compute_base_redundancy(snr)

        # Priority multipliers (learned during training)
        multipliers = {
            Priority.EMERGENCY: 2.0,  # Double redundancy
            Priority.HIGH: 1.5,
            Priority.NORMAL: 1.0,
            Priority.LOW: 0.7
        }

        return base_redundancy * multipliers[priority]

    def select_patterns(self, priority, available_patterns):
        """Priority affects pattern selection"""

        if priority == Priority.EMERGENCY:
            # Use most robust patterns
            return self.most_robust_patterns(available_patterns)
        elif priority == Priority.LOW:
            # Use efficient patterns
            return self.most_efficient_patterns(available_patterns)
        else:
            # Balanced selection
            return self.balanced_patterns(available_patterns)
```

## Queue Management

Queue management is where priority handling becomes real. When multiple messages compete for limited channel capacity, the protocol must decide who goes first. This is a discrete decision - message A before message B - that users need to understand and predict.

CASCADE uses a sophisticated queuing system that balances strict priority with fairness. Pure priority queuing can starve low-priority traffic indefinitely; pure round-robin ignores urgency. The protocol finds a middle ground that handles emergencies immediately while ensuring all traffic eventually gets through.

### Protocol Queue Ordering

The protocol layer makes discrete decisions about transmission order using a weighted priority system with starvation prevention:

Discrete decisions about transmission order:

```python
class TransmissionQueue:
    """Protocol manages queue order"""

    def __init__(self):
        self.queues = {
            Priority.EMERGENCY: deque(),
            Priority.HIGH: deque(),
            Priority.NORMAL: deque(),
            Priority.LOW: deque()
        }

    def get_next_message(self):
        """Strict priority with fairness"""

        # Always send emergency first
        if self.queues[Priority.EMERGENCY]:
            return self.queues[Priority.EMERGENCY].popleft()

        # Weighted selection for others
        weights = {
            Priority.HIGH: 0.5,
            Priority.NORMAL: 0.35,
            Priority.LOW: 0.15
        }

        # Probabilistic selection with starvation prevention
        return self.weighted_selection(weights)

    def weighted_selection(self, weights):
        """Prevent low priority starvation"""

        # Increase weight for aged messages
        for priority, queue in self.queues.items():
            if queue:
                age = time.time() - queue[0]['timestamp']
                age_boost = min(age / 3600, 2.0)  # Max 2x boost
                weights[priority] *= (1 + age_boost)

        # Select based on adjusted weights
        return self.select_by_weight(weights)
```

### Model Timing Optimization

Within each priority class, the model optimizes exact transmission timing. While the protocol determines that emergency messages go first, the model decides the optimal millisecond to start transmission, how long to make fragments, and when to expect acknowledgments:

Model decides WHEN within priority class:

```python
def optimize_transmission_timing(messages, channel_state):
    """Model optimizes timing within priority groups"""

    # Group by priority
    priority_groups = group_by_priority(messages)

    scheduled = []
    for priority, group in priority_groups.items():
        # Model decides optimal order within group
        group_schedule = model.optimize_group_timing(
            group,
            channel_state,
            priority
        )
        scheduled.extend(group_schedule)

    return scheduled
```

## Emergency Traffic Handling

Emergency traffic receives special treatment throughout the system. This isn't just higher priority - it's a different operational mode that affects authentication, queuing, relaying, and encoding. The system recognizes that when lives are at stake, normal rules must bend.

The emergency handling system is designed to work even when infrastructure fails. It doesn't require central servers, certificate authorities, or even consistent connectivity. A single emergency message can propagate through a partially connected mesh, finding its way to help.

### Immediate Relay

Emergency traffic completely bypasses normal queuing and gets immediate channel access. If the channel is busy with routine traffic, that traffic pauses. If multiple emergencies compete, they share the channel using collision avoidance patterns:

Emergency traffic bypasses normal queuing:

```python
def handle_emergency_traffic(message):
    """Emergency gets immediate attention"""

    if message['priority'] == Priority.EMERGENCY:
        # Interrupt current transmission if possible
        if can_interrupt_current():
            pause_current_transmission()

        # Immediate transmission with maximum redundancy
        emergency_encoding = model.emergency_mode(message)
        transmit_immediate(emergency_encoding)

        # Auto-relay to all known stations
        broadcast_emergency_relay(message)
```

### Authentication Relaxation

In emergencies, perfect authentication becomes secondary to message delivery. The protocol relaxes authentication requirements for emergency traffic - it's better to relay an unverified emergency that might be false than to block a real emergency due to authentication failure.

This doesn't mean emergency traffic is unauthenticated - the system still attempts verification. But authentication failure doesn't block transmission. The message is marked as unverified and operators can make informed decisions:

Emergency traffic has relaxed authentication:

```python
def verify_emergency(message):
    """Best-effort auth for emergency"""

    if message['priority'] == Priority.EMERGENCY:
        # Try to verify but don't block
        auth_status = attempt_verification(message)

        if not auth_status:
            message['unverified'] = True
            log_unverified_emergency(message)

        # Process regardless of auth
        return True
    else:
        # Normal authentication required
        return verify_normal(message)
```

## Priority Indication

Priority must be communicated implicitly through the signal itself, without requiring explicit headers that consume bandwidth. CASCADE achieves this through intelligent use of patterns and timing. Receivers can determine message priority from signal characteristics before fully decoding the message.

### In-Band Signaling

The protocol reserves certain patterns for emergency traffic. When a receiver detects these patterns, it knows to process the message immediately. This works even if the message is partially corrupted:

Pattern selection indicates priority:

```python
def priority_pattern_mapping(priority):
    """Certain patterns indicate priority"""

    # Reserve patterns 0-3 for emergency
    if priority == Priority.EMERGENCY:
        return [0, 1, 2, 3]

    # Patterns chosen by model for others
    return None  # Model decides
```

### Beacon Priority

When a station has queued messages, its beacons indicate the highest priority traffic waiting. This allows other stations to make informed decisions about channel access. A station with emergency traffic waiting will beacon differently than one with only routine messages:

Beacons indicate priority traffic waiting:

```python
def create_priority_beacon(pending_messages):
    """Beacon indicates highest priority waiting"""

    highest_priority = min(m['priority'] for m in pending_messages)

    beacon = {
        'callsign': my_callsign,
        'priority_traffic': highest_priority,
        'message_count': len(pending_messages)
    }

    if highest_priority == Priority.EMERGENCY:
        beacon['emergency'] = True

    return beacon
```

## Logging and Accountability

Priority modifications must be transparent and auditable. Every change is logged with justification, creating an audit trail that can be reviewed later. This accountability ensures the priority system isn't abused while still allowing necessary flexibility.

The logging system is designed to work offline - logs are stored locally and synchronized when connectivity permits. This ensures accountability even in disconnected emergency operations.

### Priority Change Audit

Every priority modification creates a detailed audit record that captures not just what changed but why:

All modifications are logged:

```python
class PriorityAudit:
    """Track all priority modifications"""

    def log_change(self, message_id, original, new, reason, relay_callsign):
        """Create audit record"""

        record = {
            'timestamp': time.time(),
            'message_id': message_id,
            'original_priority': original,
            'new_priority': new,
            'reason': reason,
            'relay_station': relay_callsign,
            'conditions': self.current_conditions()
        }

        # Store locally
        self.audit_log.append(record)

        # Include in next beacon
        self.pending_audit_reports.append(record)
```

## Training Considerations

Training the model for priority handling requires careful attention to realistic scenarios. The model must learn from a distribution that matches real-world usage - mostly normal traffic with occasional high priority and rare emergencies. Training on unrealistic distributions would create a model that performs poorly in deployment.

### Priority Distribution

Real networks see highly skewed priority distributions. Emergency traffic might be <1% of messages but must be handled perfectly. The training process must account for this imbalance:

Training data should reflect realistic priority distribution:

```python
def generate_training_batch():
    """Realistic priority distribution"""

    # Approximate real-world distribution
    distribution = {
        Priority.EMERGENCY: 0.01,  # 1% emergency
        Priority.HIGH: 0.09,       # 9% high priority
        Priority.NORMAL: 0.70,     # 70% normal
        Priority.LOW: 0.20         # 20% low priority
    }

    messages = []
    for _ in range(batch_size):
        priority = sample_from_distribution(distribution)
        messages.append(create_training_message(priority))

    return messages
```

### Emergency Scenario Training

Because emergencies are rare but critical, they require special training attention. The model must be explicitly trained on various emergency scenarios to ensure robust handling when they occur in reality:

Special training for emergency handling:

```python
def emergency_scenario_training():
    """Focused emergency training"""

    scenarios = [
        "multiple_emergency",      # Multiple simultaneous emergencies
        "degraded_channel",        # Emergency during poor conditions
        "relay_chain",            # Multi-hop emergency relay
        "authentication_failure"   # Unverified emergency
    ]

    for scenario in scenarios:
        batch = generate_emergency_scenario(scenario)
        loss = model.train_on_batch(batch)

        # Extra weight for emergency scenarios
        weighted_loss = loss * 5.0
        weighted_loss.backward()
```

## Benefits

The priority handling system provides essential capabilities for amateur radio networks:

1. **User Control**: Users determine importance
   - Operators retain authority over their traffic
   - No algorithmic second-guessing of intent
   - Respects amateur radio autonomy traditions

2. **Adaptive Relay**: Stations can upgrade based on content
   - Network can recognize emergencies even if sender doesn't
   - Creates resilient emergency response without central control
   - Gateway stations can help non-radio users access emergency services

3. **Fairness**: Prevents starvation of low priority
   - Weighted queuing ensures all traffic eventually transmits
   - Age-based priority boost prevents infinite delays
   - System remains usable for routine traffic even during emergencies

4. **Emergency Ready**: Immediate handling of life safety
   - Emergency traffic gets immediate channel access
   - Authentication requirements relaxed but not eliminated
   - Automatic relay propagation through mesh

5. **Accountability**: All changes are logged
   - Every priority modification is recorded with justification
   - Audit trail for post-event analysis
   - Prevents abuse while maintaining flexibility

6. **Model Optimization**: Learned resource allocation by priority
   - Discovers optimal encoding strategies for each priority level
   - Balances redundancy with efficiency based on importance
   - Continuously improves through operational experience

## Design Philosophy

Priority handling in CASCADE embodies several key principles:

**Human Judgment Matters**: The protocol trusts human operators to make priority decisions. While the system can suggest and modify priorities, humans retain ultimate control. This respects the amateur radio tradition of operator responsibility.

**Graceful Degradation**: As conditions worsen, the system naturally prioritizes more important traffic. Low priority messages might be delayed or dropped, but emergency traffic gets through. This creates natural load shedding without complex algorithms.

**Transparency Over Automation**: Every priority decision is visible and auditable. The system explains why priorities were modified, allowing operators to understand and trust the system's behavior.

**Emergency First**: When lives are at stake, normal rules bend. Authentication becomes optional, queues are bypassed, and maximum redundancy is applied. The system recognizes that false positives are better than false negatives in emergencies.

## Real-World Impact

In operational deployment, effective priority handling can literally save lives. During natural disasters, CASCADE networks naturally prioritize evacuation orders over routine check-ins. Medical emergencies automatically get precedence over administrative traffic.

The ability to modify priority during relay is particularly powerful. A message saying "feeling sick" might be sent as normal priority, but if a relay station recognizes symptoms of carbon monoxide poisoning, it can upgrade to emergency. This distributed intelligence makes the network more capable than any individual station.

For emergency coordinators, the priority system provides predictable behavior during chaotic events. They know emergency traffic will get through, even if routine reports are delayed. This confidence allows them to rely on CASCADE for critical communications.