# CASCADE Modem - System Architecture

This document describes the architecture of the CASCADE modem web application.

---

## Overview

CASCADE modem is a **web-based digital modem** for HF radio, built as:
- **Backend:** Python server (FastAPI + PyTorch)
- **Frontend:** React web application (TypeScript + shadcn/ui)
- **Communication:** Native WebSocket (real-time bidirectional)

---

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (User)                           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              React Frontend (TypeScript)                    │ │
│  │                                                             │ │
│  │  Components:                                                │ │
│  │  ├─ Radio Control (frequency, mode, PTT)                   │ │
│  │  ├─ Conversation List (multi-user threads)                 │ │
│  │  ├─ Network Topology (D3.js graph, 1024 users)            │ │
│  │  ├─ Net Operations (controller/member/relay UI)           │ │
│  │  ├─ Kernel Status (cache, validity, available tones)      │ │
│  │  └─ Emergency Modal (relay approval)                       │ │
│  │                                                             │ │
│  │  State: Context API + useReducer                           │ │
│  │  Communication: Native WebSocket                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             ↕ WebSocket                          │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    Python Backend Server                         │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  FastAPI Server                             │ │
│  │                                                             │ │
│  │  REST API:                                                  │ │
│  │  ├─ GET/POST /api/config                                   │ │
│  │  ├─ GET /api/health                                        │ │
│  │  ├─ GET /api/network/topology                              │ │
│  │  └─ GET /api/kernels                                       │ │
│  │                                                             │ │
│  │  WebSocket: /ws                                            │ │
│  │  ├─ Message decode stream (real-time)                     │ │
│  │  ├─ Command handling (send, freq change, net ops)         │ │
│  │  └─ Broadcast to all clients                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↕                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  Protocol Layer                             │ │
│  │                                                             │ │
│  │  ├─ Message Format (binary: 19 + payload + 8 bytes)       │ │
│  │  ├─ Validation (CRC32 + xxHash32 dual-layer)              │ │
│  │  ├─ Kernel Lifecycle (3-round: pro/anti/adapt)            │ │
│  │  ├─ Pattern Assignment (pool management, rotation)         │ │
│  │  ├─ Net Operations (controller, member, relay logic)      │ │
│  │  ├─ Emergency Relay (manual approval, multi-hop)          │ │
│  │  └─ Mesh Topology (hash-based discovery)                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↕                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   Model Layer (PyTorch)                     │ │
│  │                                                             │ │
│  │  ├─ Pattern Correlation (128 patterns, 78-tone grid)      │ │
│  │  ├─ Multi-User Decode (up to 45 simultaneous)             │ │
│  │  ├─ Kernel Generation (RX optimization, 64-bit)           │ │
│  │  ├─ Adaptive Modulation (BPSK/QPSK/8-QAM selection)       │ │
│  │  └─ Encoding (message → audio signal)                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↕                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                 Hardware Layer                              │ │
│  │                                                             │ │
│  │  ├─ Radio Control (Hamlib via ctypes)                     │ │
│  │  │   - Set/get frequency                                  │ │
│  │  │   - Set mode (USB/LSB)                                 │ │
│  │  │   - PTT control                                        │ │
│  │  │                                                         │ │
│  │  └─ Audio I/O (sounddevice)                               │ │
│  │      - Receive audio stream (real-time)                   │ │
│  │      - Transmit audio queue                               │ │
│  │      - FFT for spectrum display                           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                       Hardware                                   │
│                                                                  │
│  ├─ Radio (via Hamlib: 100+ models)                            │
│  └─ Audio Interface (sound card)                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Receive Path (Decoding)

```
Audio In → sounddevice → FFT → PyTorch Model → Protocol Validation
  →  WebSocket → React UI → User sees decoded message

Detailed:
1. Sound card captures audio (12 kHz sample rate)
2. sounddevice streams audio to backend
3. FFT converts to frequency domain
4. PyTorch model correlates against 128 patterns (45 simultaneous)
5. Decoded symbols → binary message reconstruction
6. Protocol layer validates (CRC32 + xxHash32)
7. Parse message (from/to/content/priority)
8. Broadcast via WebSocket to all connected clients
9. React updates conversation list
10. User sees message in UI
```

### Transmit Path (Encoding)

```
User types message → React → WebSocket → Protocol Format → PyTorch Model
  → Audio Out → sounddevice → Radio

Detailed:
1. User composes message in React UI
2. Frontend sends via WebSocket ("send_message" event)
3. Backend receives, formats binary message (19 + payload + 8)
4. Compute CRC32 + xxHash32
5. Look up destination kernel (RX optimization)
6. PyTorch model encodes:
   - Select patterns from assigned pool
   - Choose modulation (BPSK/QPSK/8-QAM based on kernel)
   - Generate audio waveform (1.6s pattern duration)
7. Queue audio for transmission
8. Set PTT ON via Hamlib
9. sounddevice plays audio to radio
10. Set PTT OFF when complete
11. Await ACK from destination
```

### Kernel Exchange Flow

```
Station A → Broadcast RX Kernel → WebSocket → All Stations Cache

When B transmits to A:
1. B looks up A's RX kernel from cache
2. B uses A's kernel as encoder hints:
   - Modulation preference (QPSK vs 8-QAM)
   - Available tones (which of 78 tones A can decode)
   - Hardware tier (capacity, FEC needs)
3. B encodes message optimized FOR A's receiver
4. A decodes more reliably (kernel helped!)
5. A sends ACK with own RX kernel (refreshed)
6. B updates cache

3-Round Exchange (if interference):
1. Pro: A broadcasts RX kernel
2. Anti: B responds with antikernel (interference report)
3. Adapt: A adapts RX kernel to avoid interfering with B
```

---

## State Management (Frontend)

### React Context + useReducer

```typescript
// Global state structure
interface CASCADEState {
  // Connection
  connected: boolean;
  sessionId: string | null;

  // Radio
  radio: {
    connected: boolean;
    frequency: number;      // Hz
    mode: string;           // USB/LSB
    ptt: boolean;           // Transmitting?
  };

  // Audio
  audio: {
    running: boolean;
    level: number;          // 0-100
  };

  // Network
  network: {
    activeUsers: number;    // Currently transmitting (up to 45)
    totalCapacity: number;  // 1024 (via pattern reuse)
    myPatterns: number[];   // Assigned pattern pool (8 patterns)
    kernelsCached: number;  // How many kernels cached
  };

  // Conversations
  conversations: Map<string, {
    messages: Message[];
    lastSNR: number;
    kernelValid: boolean;
  }>;

  // Current Net
  currentNet: NetState | null;

  // Profile
  profile: {
    callsign: string;
    gridSquare: string;
    hardwareTier: string;
  };
}

// Actions dispatched to reducer
type Action =
  | { type: 'CONNECT'; payload: { sessionId: string } }
  | { type: 'UPDATE_RADIO'; payload: Partial<RadioState> }
  | { type: 'MESSAGE_RECEIVED'; payload: Message }
  | { type: 'NET_FORMED'; payload: NetState }
  | ...
```

**Benefits:**
- No external dependencies (built into React)
- Clear action types (easy debugging)
- Predictable state updates
- Works well with TypeScript

---

## WebSocket Protocol

### Message Format (JSON over WebSocket)

**Client → Server:**
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

**Server → Client:**
```json
{
  "type": "message_decoded",
  "data": {
    "from": "W2DEF",
    "to": "W1ABC",
    "content": "Hello CASCADE",
    "snr": 10,
    "timestamp": 1234567890.123
  }
}
```

**Event Types:**

| Type | Direction | Description |
|------|-----------|-------------|
| `connect_ack` | Server → Client | Connection acknowledged |
| `network_state` | Server → Client | Network topology update |
| `message_decoded` | Server → Client | Decoded message (from chaos) |
| `kernel_update` | Server → Client | New kernel received |
| `emergency_relay_request` | Server → Client | Emergency needs relay |
| `send_message` | Client → Server | User wants to send |
| `set_frequency` | Client → Server | Change radio frequency |
| `form_net` | Client → Server | Create new net |
| `request_slot` | Client → Server | Request net slot |

See [API.md](API.md) for complete protocol reference.

---

## Phase Architecture

CASCADE is being built in **7 phases** over 14 weeks:

### Phase 1: Foundation ✅ (Weeks 1-2)
**Goal:** Radio control from browser, spectrum display

Components:
- FastAPI server + native WebSocket
- React frontend with shadcn/ui
- Hamlib radio control (mock mode)
- Audio I/O basics
- FFT spectrum display

### Phase 2: Single-User (Weeks 3-4)
**Goal:** Send/receive one message

Components:
- Binary message format (19 + payload + 8)
- CRC32 + xxHash32 validation
- PyTorch model (single user decode)
- Message composer UI
- Conversation display

### Phase 3: Kernel System (Weeks 5-6)
**Goal:** RX optimization, adaptive encoding

Components:
- Kernel generation (64-bit RX hints)
- 3-round exchange (pro/anti/adapt)
- Available tone encoding (40-bit run-length)
- Kernel cache management
- Adaptive modulation (BPSK/QPSK/8-QAM)

### Phase 4: Multi-User Chaos (Weeks 7-8)
**Goal:** Decode 45 simultaneous users

Components:
- Multi-user pattern correlation
- 45-user decode capacity
- Conversation threading
- Conversation list UI
- Pattern activity visualization

### Phase 5: Net Operations (Weeks 9-10)
**Goal:** Net controller, slots, relays

Components:
- Net formation (controller/member/relay)
- Net profiles (DX/Ragchew/Emergency/Contest)
- Slot scheduling (pre-encoded durations)
- Network topology graph (D3.js)
- Relay assignment algorithm

### Phase 6: Emergency Relay (Weeks 11-12)
**Goal:** Emergency relay network

Components:
- Emergency detection
- Manual relay approval UI
- Multi-hop tracking
- Rate limiting (5 msg/hour)
- Progressive compression

### Phase 7: Production (Weeks 13-14)
**Goal:** Deployment-ready system

Components:
- Telemetry collection (neural state)
- Deployment scripts (systemd, Docker)
- CI/CD pipeline (GitHub Actions)
- Performance optimization
- Complete documentation

---

## Technology Decisions

### Why Native WebSockets?

**Chose:** Native WebSocket API
**Instead of:** socket.io

**Reasons:**
- ✅ Built into browsers (no client library)
- ✅ Built into FastAPI (no python-socketio)
- ✅ Simpler protocol (JSON over WebSocket)
- ✅ Lighter bundle size
- ✅ Better DevTools support
- ✅ Faster (less overhead)

### Why Context + useReducer?

**Chose:** React Context API + useReducer
**Instead of:** Zustand, Redux

**Reasons:**
- ✅ No external dependency
- ✅ Built into React (well-documented)
- ✅ Sufficient for CASCADE's complexity
- ✅ Works great with TypeScript
- ✅ React DevTools shows state natively

### Why shadcn/ui?

**Chose:** shadcn/ui + Tailwind
**Instead of:** Material-UI, Ant Design

**Reasons:**
- ✅ Clean, readable HTML output
- ✅ You own the components (copy into project)
- ✅ Built on Radix UI (excellent accessibility)
- ✅ Tailwind for styling (readable classes)
- ✅ TypeScript-first
- ✅ Easy to customize

### Why PyTorch?

**Chose:** PyTorch
**Instead of:** TensorFlow, ONNX Runtime

**Reasons:**
- ✅ CASCADE model already trained in PyTorch
- ✅ Python-native (easy integration)
- ✅ Good Raspberry Pi support
- ✅ Can export to ONNX later if needed

---

## Performance Considerations

### Backend

**Target:** Raspberry Pi 4 (4GB RAM)

**Performance requirements:**
- PyTorch inference: <10ms per decode (8.7ms achieved)
- Multi-user decode: 45 simultaneous users
- WebSocket latency: <50ms
- Audio buffering: <100ms

**Optimizations:**
- Use PyTorch JIT (torchscript) for faster inference
- Audio processing on separate thread
- WebSocket broadcast async (non-blocking)
- Pattern correlation parallelized

### Frontend

**Target:** Modern browsers (desktop, tablet, mobile)

**Performance requirements:**
- First load: <2s
- WebSocket reconnect: <1s
- UI updates: 60 FPS
- Bundle size: <500KB (gzipped)

**Optimizations:**
- Vite code splitting (lazy load routes)
- shadcn/ui tree-shaking (only import used components)
- D3.js canvas rendering (not SVG for large graphs)
- React.memo for expensive components
- useCallback/useMemo where appropriate

---

## Security Considerations

### WebSocket Security

- Use `wss://` (WebSocket Secure) in production
- Validate all incoming messages
- Rate limiting on message sends
- Authenticate clients (future: JWT tokens)

### Message Validation

**Dual-layer validation prevents neural network hallucinations:**

1. **CRC32** - Error detection
   - NN learns to predict this (improves training)
   - Fast to compute

2. **xxHash32** - Validity checking
   - NN cannot forge (prevents false positives)
   - Ensures message is real, not hallucination

### Privacy

- Callsigns hashed in binary protocol (32-bit hash)
- Full callsign only in 4-FSK beacons (FCC ID requirement)
- No GPS coordinates stored (grid square only)
- Telemetry anonymized

---

## Deployment Architecture

### Development

```
Developer Machine:
├─ Backend (localhost:8000)
│   - Mock radio/audio
│   - Auto-reload on code changes
└─ Frontend (localhost:5173)
    - Vite dev server
    - Hot module replacement (HMR)
```

### Production (Single Machine)

```
Shack Computer (Linux/Windows/Mac):
├─ Backend (systemd service)
│   - Real Hamlib connection
│   - Real audio interface
│   - Serves frontend static files at /
└─ Access: http://localhost:8000 (or LAN IP)
```

### Production (Headless Shack)

```
Shack: Raspberry Pi (headless)
├─ Backend running (systemd)
├─ Radio + sound card connected
└─ Network: 192.168.1.100:8000

Access from:
├─ Desktop browser: http://192.168.1.100:8000
├─ Tablet browser: http://192.168.1.100:8000
└─ Phone browser: http://192.168.1.100:8000
```

---

## Future Enhancements

### Phase 8+ (Post-MVP)

- **Cloud Telemetry Dashboard**: Global CASCADE network visualization
- **ONNX Runtime**: Export model for C++ deployment (2-3x faster)
- **Rust Protocol Layer**: Memory safety + performance
- **Multi-Language Support**: Internationalization (i18n)
- **Voice Integration**: Web Audio API for voice modes
- **Advanced Visualizations**: 3D network topology, animated waterfall

---

## See Also

- **[README.md](README.md)** - Setup and usage guide
- **[API.md](API.md)** - REST + WebSocket protocol reference
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guide
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment

### CASCADE Protocol Documentation

Main CASCADE docs (in `/docs/`):
- `architecture.md` - 128-pattern chaos system
- `protocol/` - Binary message format, kernels, nets
- `model/` - Neural network architecture details

---

**Last updated:** 2025-10-04
