# CASCADE Modem - API Reference

Complete reference for REST API endpoints and WebSocket protocol.

---

## Base URL

**Development:** `http://localhost:8000`
**Production:** `http://YOUR_SERVER_IP:8000`

---

## REST API Endpoints

### Health & Status

#### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "radio_connected": true,
  "audio_running": true,
  "active_users": 12
}
```

#### GET /

Root endpoint (API information).

**Response:**
```json
{
  "service": "CASCADE Modem Server",
  "version": "0.1.0",
  "callsign": "W1ABC",
  "status": "running"
}
```

---

### Configuration

#### GET /api/config

Get current configuration.

**Response:**
```json
{
  "callsign": "W1ABC",
  "gridSquare": "FN42mc",
  "frequency": 14074000,
  "mode": "USB",
  "hardwareTier": "rpi4",
  "maxSimultaneousUsers": 15
}
```

#### POST /api/config

Update configuration.

**Request:**
```json
{
  "callsign": "W1ABC",
  "gridSquare": "FN42mc",
  "hardwareTier": "desktop"
}
```

**Response:**
```json
{
  "status": "updated"
}
```

---

### Radio Control

#### GET /api/radio/status

Get radio status.

**Response:**
```json
{
  "connected": true,
  "frequency": 14074000,
  "mode": "USB",
  "ptt": false,
  "model": 3037,
  "modelName": "Icom IC-7300"
}
```

#### POST /api/radio/frequency

Set operating frequency.

**Request:**
```json
{
  "frequency": 14074000
}
```

**Response:**
```json
{
  "status": "ok",
  "frequency": 14074000
}
```

#### POST /api/radio/mode

Set operating mode.

**Request:**
```json
{
  "mode": "USB",
  "bandwidth": 3000
}
```

**Response:**
```json
{
  "status": "ok",
  "mode": "USB",
  "bandwidth": 3000
}
```

#### GET /api/radio/supported

List supported radio models (Hamlib).

**Response:**
```json
{
  "radios": {
    "IC-7300": 3037,
    "IC-7610": 3073,
    "FT-991A": 1035,
    "FT-891": 1041,
    "TS-590SG": 2045,
    "Dummy": 1
  }
}
```

---

### Audio

#### GET /api/audio/devices

List audio devices.

**Response:**
```json
{
  "devices": [
    {
      "index": 0,
      "name": "Built-in Microphone",
      "channels": 2,
      "sampleRate": 44100,
      "type": "input"
    },
    {
      "index": 1,
      "name": "USB Audio Interface",
      "channels": 2,
      "sampleRate": 48000,
      "type": "input"
    }
  ]
}
```

#### POST /api/audio/device

Select audio device.

**Request:**
```json
{
  "inputDevice": 1,
  "outputDevice": 2
}
```

**Response:**
```json
{
  "status": "ok",
  "inputDevice": 1,
  "outputDevice": 2
}
```

#### GET /api/audio/levels

Get audio levels (RX/TX).

**Response:**
```json
{
  "rxLevel": 45,
  "txLevel": 0,
  "peak": 67
}
```

---

### Messages

#### GET /api/messages

Get recent message history.

**Query Parameters:**
- `limit` (optional): Number of messages (default: 50)
- `station` (optional): Filter by station callsign

**Response:**
```json
{
  "messages": [
    {
      "from": "W2DEF",
      "to": "W1ABC",
      "content": "Hello CASCADE",
      "snr": 10,
      "timestamp": 1234567890.123,
      "messageId": 12345
    }
  ]
}
```

#### POST /api/messages

Send message (alternative to WebSocket).

**Request:**
```json
{
  "to": "W2DEF",
  "content": "Hello CASCADE",
  "priority": "NORMAL"
}
```

**Response:**
```json
{
  "status": "queued",
  "messageId": 12346
}
```

#### GET /api/messages/:id

Get specific message by ID.

**Response:**
```json
{
  "from": "W2DEF",
  "to": "W1ABC",
  "content": "Hello CASCADE",
  "snr": 10,
  "timestamp": 1234567890.123,
  "messageId": 12345,
  "patterns": [5, 19],
  "validated": true
}
```

---

### Network Topology

#### GET /api/network/topology

Get current network graph.

**Response:**
```json
{
  "nodes": [
    {
      "callsign": "W1ABC",
      "gridSquare": "FN42mc",
      "snr": 15,
      "lastHeard": 1234567890.123,
      "patterns": [5, 12, 19, 27]
    }
  ],
  "links": [
    {
      "from": "W1ABC",
      "to": "W2DEF",
      "snr": 12,
      "relayVia": null
    }
  ],
  "activeUsers": 12,
  "totalCapacity": 1024
}
```

#### GET /api/network/stations

Get list of known stations.

**Response:**
```json
{
  "stations": [
    {
      "callsign": "W2DEF",
      "gridSquare": "FN31pr",
      "lastHeard": 1234567890.123,
      "snr": 10,
      "kernelCached": true,
      "kernelAge": 120
    }
  ]
}
```

---

### Kernels

#### GET /api/kernels

Get all cached kernels.

**Response:**
```json
{
  "kernels": [
    {
      "station": "W2DEF",
      "kernelData": "base64_encoded_64_bits",
      "validUntil": 1234567890.123,
      "age": 120,
      "availableTones": [0, 1, 2, 35, 36, 77],
      "modulation": "QPSK",
      "hardwareTier": "rpi4"
    }
  ]
}
```

#### GET /api/kernels/:station

Get kernel for specific station.

**Response:**
```json
{
  "station": "W2DEF",
  "kernelData": "base64_encoded_64_bits",
  "validUntil": 1234567890.123,
  "age": 120,
  "availableTones": [0, 1, 2, 35, 36, 77],
  "modulation": "QPSK",
  "hardwareTier": "rpi4"
}
```

#### POST /api/kernels/refresh

Force kernel refresh for station.

**Request:**
```json
{
  "station": "W2DEF"
}
```

**Response:**
```json
{
  "status": "requested",
  "station": "W2DEF"
}
```

---

### Net Operations

#### GET /api/nets

Get active nets.

**Response:**
```json
{
  "nets": [
    {
      "netId": "net_12345",
      "controller": "W1NET",
      "profile": "DX",
      "purpose": "20m DX Net",
      "members": 15,
      "startTime": 1234567890.123
    }
  ]
}
```

#### POST /api/nets

Form new net (as controller).

**Request:**
```json
{
  "profile": "DX",
  "purpose": "20m DX Net",
  "maxMessageBytes": 32,
  "maxSlotSeconds": 2.0
}
```

**Response:**
```json
{
  "status": "formed",
  "netId": "net_12345",
  "controller": "W1ABC"
}
```

#### GET /api/nets/:id

Get net details.

**Response:**
```json
{
  "netId": "net_12345",
  "controller": "W1NET",
  "profile": "DX",
  "purpose": "20m DX Net",
  "members": [
    {
      "callsign": "VK2ZOI",
      "gridSquare": "QF56",
      "relay": "K0BB",
      "patterns": [12, 15]
    }
  ],
  "relays": ["K0BB", "N7XYZ"],
  "slotSchedule": []
}
```

#### POST /api/nets/:id/join

Join net as member.

**Request:**
```json
{
  "capabilities": "rpi4"
}
```

**Response:**
```json
{
  "status": "joined",
  "netId": "net_12345",
  "assignedRelay": "K0BB"
}
```

#### POST /api/nets/:id/slots

Request slot (member) or assign slots (controller).

**Request (member):**
```json
{
  "messageText": "Hello from VK2ZOI",
  "exactDuration": 1.6
}
```

**Request (controller):**
```json
{
  "slots": [
    {
      "speaker": "VK2ZOI",
      "startTime": 1234567890.123,
      "duration": 1.6,
      "pattern": 12
    }
  ]
}
```

**Response:**
```json
{
  "status": "scheduled",
  "yourSlot": {
    "startTime": 1234567890.123,
    "duration": 1.6,
    "pattern": 12
  }
}
```

---

### Telemetry (Phase 7)

#### POST /api/telemetry/upload

Upload telemetry to cloud.

**Request:**
```json
{
  "neuralState": "base64_encoded_state",
  "metadata": {
    "callsign": "W1ABC",
    "timestamp": 1234567890.123
  }
}
```

**Response:**
```json
{
  "status": "uploaded",
  "telemetryId": "telem_12345"
}
```

#### GET /api/telemetry/stats

Get local telemetry statistics.

**Response:**
```json
{
  "messagesDecoded": 1234,
  "messagesSent": 567,
  "avgSNR": 12.5,
  "uplinkSuccess": 0.95
}
```

---

## WebSocket Protocol

### Connection

**Endpoint:** `ws://localhost:8000/ws`

**Connect:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('Connected to CASCADE server');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message.type, message.data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from CASCADE server');
};
```

---

### Message Format

All WebSocket messages are JSON:

```json
{
  "type": "message_type",
  "data": { ... }
}
```

---

### Server → Client Events

#### connect_ack

Connection acknowledged.

**Payload:**
```json
{
  "type": "connect_ack",
  "data": {
    "sessionId": "session_12345",
    "cascadeVersion": "0.1.0",
    "callsign": "W1ABC",
    "gridSquare": "FN42mc"
  }
}
```

#### network_state

Network topology update.

**Payload:**
```json
{
  "type": "network_state",
  "data": {
    "activeUsers": 12,
    "totalCapacity": 1024,
    "myPatterns": [5, 12, 19, 27],
    "kernelsCached": 34,
    "currentNet": null,
    "relayMode": false
  }
}
```

#### message_decoded

Decoded message from multi-user chaos.

**Payload:**
```json
{
  "type": "message_decoded",
  "data": {
    "from": "W2DEF",
    "to": "W1ABC",
    "content": "Hello CASCADE",
    "messageId": 12345,
    "priority": "NORMAL",
    "snr": 10,
    "patternsUsed": [5, 19],
    "relayedVia": null,
    "timestamp": 1234567890.123
  }
}
```

#### kernel_update

New kernel received.

**Payload:**
```json
{
  "type": "kernel_update",
  "data": {
    "fromStation": "W2DEF",
    "kernelData": "base64_encoded_64_bits",
    "kernelType": "standard",
    "validUntil": 1234567890.123,
    "availableTones": [0, 1, 2, 35, 36, 77]
  }
}
```

#### emergency_relay_request

Emergency message needs relay.

**Payload:**
```json
{
  "type": "emergency_relay_request",
  "data": {
    "originStation": "W3XYZ",
    "destinationStation": "W1EOC",
    "message": "Medical emergency at grid FN42mc",
    "messageId": 911,
    "hopCount": 1,
    "approvalRequired": true
  }
}
```

#### net_formed

New net created.

**Payload:**
```json
{
  "type": "net_formed",
  "data": {
    "netId": "net_12345",
    "controller": "W1NET",
    "profile": "DX",
    "purpose": "20m DX Net"
  }
}
```

#### net_slot_assigned

Slot assigned in net.

**Payload:**
```json
{
  "type": "net_slot_assigned",
  "data": {
    "speaker": "VK2ZOI",
    "startTime": 1234567890.123,
    "duration": 1.6,
    "pattern": 12
  }
}
```

#### net_closed

Net closed.

**Payload:**
```json
{
  "type": "net_closed",
  "data": {
    "netId": "net_12345",
    "duration": 1800
  }
}
```

#### spectrum_data

FFT spectrum data for waterfall.

**Payload:**
```json
{
  "type": "spectrum_data",
  "data": {
    "fftData": [0.1, 0.2, 0.3, ...],
    "timestamp": 1234567890.123,
    "centerFreq": 14074000
  }
}
```

#### pattern_activity

Which patterns active in last second.

**Payload:**
```json
{
  "type": "pattern_activity",
  "data": {
    "activePatterns": [5, 12, 19, 27],
    "patternSNR": {
      "5": 12,
      "12": 8,
      "19": 15,
      "27": 10
    }
  }
}
```

#### frequency_changed

Frequency changed (broadcast).

**Payload:**
```json
{
  "type": "frequency_changed",
  "data": {
    "frequency": 14074000,
    "status": "ok"
  }
}
```

---

### Client → Server Events

#### send_message

Send message to another station.

**Payload:**
```json
{
  "type": "send_message",
  "data": {
    "to": "W2DEF",
    "content": "Hello CASCADE",
    "priority": "NORMAL"
  }
}
```

#### set_frequency

Change radio frequency.

**Payload:**
```json
{
  "type": "set_frequency",
  "data": {
    "frequency": 14074000
  }
}
```

#### set_mode

Change radio mode.

**Payload:**
```json
{
  "type": "set_mode",
  "data": {
    "mode": "USB",
    "bandwidth": 3000
  }
}
```

#### form_net

Form new net (as controller).

**Payload:**
```json
{
  "type": "form_net",
  "data": {
    "profile": "DX",
    "purpose": "20m DX Net"
  }
}
```

#### join_net

Join net as member.

**Payload:**
```json
{
  "type": "join_net",
  "data": {
    "netId": "net_12345",
    "capabilities": "rpi4"
  }
}
```

#### request_slot

Request slot in net.

**Payload:**
```json
{
  "type": "request_slot",
  "data": {
    "netId": "net_12345",
    "messageText": "Hello from VK2ZOI",
    "exactDuration": 1.6
  }
}
```

#### assign_slots

Assign slots (net controller only).

**Payload:**
```json
{
  "type": "assign_slots",
  "data": {
    "netId": "net_12345",
    "slots": [...]
  }
}
```

#### approve_relay

Approve or deny emergency relay.

**Payload:**
```json
{
  "type": "approve_relay",
  "data": {
    "messageId": 911,
    "approved": true
  }
}
```

#### update_profile

Update user profile.

**Payload:**
```json
{
  "type": "update_profile",
  "data": {
    "callsign": "W1ABC",
    "gridSquare": "FN42mc",
    "hardwareTier": "desktop"
  }
}
```

---

## Binary Message Format (Over-the-Air)

Actual CASCADE radio protocol (transmitted via audio):

```
┌───────────────────────────────────────────────┐
│        CASCADE Binary Message                  │
├───────────────────────────────────────────────┤
│  HEADER (19 bytes):                           │
│    from_hash: uint32 (4 bytes)                │
│    to_hash: uint32 (4 bytes)                  │
│    message_id: uint64 (8 bytes)               │
│    priority: uint8 (1 byte)                   │
│    payload_length: uint16 (2 bytes)           │
│                                               │
│  PAYLOAD (0-256 bytes):                       │
│    UTF-8 encoded text                         │
│                                               │
│  VALIDATION (8 bytes):                        │
│    crc32: uint32 (4 bytes)                    │
│    xxhash32: uint32 (4 bytes)                 │
└───────────────────────────────────────────────┘

Total: 27 bytes overhead + payload
Example: 128-byte payload = 155 bytes total
```

**Encoding:** Little-endian throughout

See CASCADE protocol docs in `/docs/protocol/message_format.md` for complete specification.

---

## Kernel Format

**64-bit standard kernel:**

```python
{
  'version': 5 bits,                  # CASCADE version
  'estimated_valid_seconds': 6 bits,  # 0-63 × 10s
  'confidence': 2 bits,               # Validity confidence
  'adapted_from_count': 2 bits,       # Anti-kernels incorporated
  'modulation_pref': 3 bits,          # BPSK/QPSK/8-QAM
  'hardware_tier': 2 bits,            # RPi/Desktop/GPU
  'capacity_users': 5 bits,           # 0-31 simultaneous
  'snr_floor': 5 bits,                # -24 to +8 dB
  'interference_map': 6 bits,         # Coarse interference
  'frequency_pref': 6 bits,           # 64 bins
  'timing_offset': 4 bits,            # Clock sync
  'noise_floor': 4 bits,              # Noise floor
  'power_request': 3 bits,            # TX power hint
  'qso_active': 1 bit,                # In QSO?
  'qso_partner_pattern': 6 bits,      # Partner's pattern
  'net_active': 1 bit,                # In net?
  'net_role': 2 bits,                 # Member/Relay/Controller
  'my_pattern': 6 bits                # My current pattern
}
# Total: 64 bits = 8 bytes
```

See CASCADE protocol docs in `/docs/protocol/kernel_lifecycle.md` for complete specification.

---

## Error Codes

### HTTP Error Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | OK | Successful request |
| 400 | Bad Request | Invalid parameters |
| 404 | Not Found | Endpoint doesn't exist |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Radio/audio not connected |

### WebSocket Close Codes

| Code | Meaning |
|------|---------|
| 1000 | Normal closure |
| 1001 | Going away (server shutdown) |
| 1006 | Abnormal closure (connection lost) |
| 1011 | Server error |

---

## Rate Limiting

- **Message sends:** 60 per minute per client
- **Emergency messages:** 5 per hour per station
- **Net operations:** 10 per minute per client

---

## See Also

- **[README.md](README.md)** - Setup and usage guide
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guide
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment

### CASCADE Protocol

Main CASCADE docs (in `/docs/protocol/`):
- `message_format.md` - Binary over-the-air format
- `kernel_lifecycle.md` - Kernel exchange protocol
- `net_operations.md` - Net coordination details

---

**Last updated:** 2025-10-04
