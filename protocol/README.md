# Protocol Layer - Discrete Decisions

The protocol layer handles all discrete decisions in CASCADE. These are choices that cannot be optimized through gradient descent and require explicit rules.

## Responsibilities

### WHO - Identity and Access
- Callsign management
- Pattern pool assignment (which users get which patterns)
- User authentication state

### WHETHER - Binary Decisions
- Relay approval (forward or not)
- Trust transitions (TOTP verified or not)
- Emergency override activation

### WHAT - Discrete Classifications
- Message priority levels (EMERGENCY/HIGH/NORMAL/LOW)
- Hash exchange content (which stations to share)
- ACK information (success/failure, SNR bucket)

## Key Components

### Message Format
Simple 5-field structure:
```python
{
    'from': 'W1ABC',      # Sender callsign
    'to': 'W2DEF',        # Destination
    'id': 12345,          # Message ID
    'priority': 'NORMAL', # Priority level
    'data': 'Hello'       # Content
}
```

### Pattern Pool Assignment
- Patterns 0-3: Emergency/broadcast reserved
- Patterns 4-63: Dynamically assigned
- 8-16 patterns per active user
- Rotation every 100 transmissions

### Hash Exchange
- Callsign-based hashes (no salting)
- SNR-scaled exchange frequency
- 15-minute memory window
- Enables distributed mesh discovery

### Kernel Hint Routing
- Pairwise hint exchange
- Receiver generates, transmitter uses
- 10-minute expiration
- Improves weak link performance

### Trust State Machine
```
UNTRUSTED → TOTP_TRUSTED → HMAC_ALLOWED
```
Transitions based on verification and link quality.

### ACK System
- 4-bit coarse SNR (-20, -10, 0, +10 dB)
- Between-frame transmission
- Pattern success feedback
- Optional kernel hints

## Interface with Model

The protocol provides constraints to the model:
- Assigned pattern pool
- Priority weight
- Time constraints
- Target destination

The model returns optimizations within these constraints.