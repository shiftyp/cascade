# CASCADE Modem - Python Server + React WebApp

Web-based digital modem for CASCADE HF radio protocol.

**Architecture:** Python backend (FastAPI + PyTorch) + React frontend (TypeScript + shadcn/ui)

**Communication:** Multi-user chaos coordination with kernel-driven adaptive encoding

---

## Features

- 🔊 **Audio I/O** via sounddevice (cross-platform)
- 📻 **Radio Control** via Hamlib (100+ radio models supported)
- 🧠 **Neural Network** PyTorch model (decode up to 45 simultaneous users)
- 🌐 **Native WebSocket** real-time bidirectional communication
- 📱 **Responsive UI** works on desktop, tablet, mobile
- 🎨 **Clean HTML** shadcn/ui components with Tailwind CSS
- 🔌 **Mock Mode** develop without radio hardware
- 🧪 **Well-Tested** pytest (backend), vitest (frontend), Playwright (E2E)

---

## Quick Start

### Backend (Python Server)

```bash
cd modules/modem/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
# Server runs on http://localhost:8000
```

### Frontend (React WebApp)

```bash
cd modules/modem/frontend

# Install dependencies
npm install

# Start dev server
npm run dev
# Opens http://localhost:5173
```

### Access Application

Open browser to **http://localhost:5173** (frontend proxies API requests to backend)

---

## Configuration

Create `.env` file in `backend/`:

```env
# User Profile
CALLSIGN=W1ABC
GRID_SQUARE=FN42mc
HARDWARE_TIER=rpi4

# Radio Settings (Hamlib)
RADIO_MODEL=3037          # Hamlib model number (3037 = Icom IC-7300)
RADIO_PORT=/dev/ttyUSB0   # Serial port (COM3 on Windows)
RADIO_BAUD=9600          # CAT baud rate

# Frequency & Mode
FREQUENCY=14074000        # Operating frequency in Hz (14.074 MHz = 20m)
MODE=USB                  # Operating mode
BANDWIDTH=3000           # Filter bandwidth in Hz

# Audio Settings
SAMPLE_RATE=12000        # Audio sample rate (Hz)
# AUDIO_INPUT_DEVICE=0   # Optional: specify input device index
# AUDIO_OUTPUT_DEVICE=0  # Optional: specify output device index

# CASCADE Protocol
MAX_SIMULTANEOUS_USERS=15  # Decode capacity (RPi4: 15, Desktop: 30-50)

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Model (optional, for future phases)
# MODEL_PATH=/path/to/cascade_model.pt

# Telemetry (optional, Phase 7)
# TELEMETRY_ENABLED=false
# TELEMETRY_ENDPOINT=https://api.cascade.example.com/telemetry
```

### Supported Radio Models

Common Hamlib model numbers:

| Radio | Model Number |
|-------|--------------|
| Icom IC-7300 | 3037 |
| Icom IC-7610 | 3073 |
| Yaesu FT-991A | 1035 |
| Yaesu FT-891 | 1041 |
| Kenwood TS-590SG | 2045 |
| Dummy (testing) | 1 |

Find your radio: https://github.com/Hamlib/Hamlib/wiki/Supported-Radios

---

## Development

### Mock Mode (No Hardware Required)

Backend automatically uses mock mode if Hamlib library not found:

```bash
# Backend runs in mock mode (simulated radio/audio)
python main.py

# Frontend works normally
npm run dev
```

Mock mode simulates:
- Radio frequency/mode changes
- PTT control
- Audio input (test signals)
- Decoded messages

### Running Tests

**Backend:**
```bash
cd backend
pytest                    # All tests
pytest -v                 # Verbose
pytest --cov=.           # With coverage
pytest tests/test_api/   # Specific module
```

**Frontend:**
```bash
cd frontend
npm test                 # Unit tests (vitest)
npm run test:coverage    # With coverage
npm run test:e2e        # End-to-end (Playwright)
```

### Code Quality

**Backend:**
```bash
# Linting
ruff check .

# Formatting
black .

# Type checking (if using mypy)
mypy .
```

**Frontend:**
```bash
# Linting
npm run lint

# Formatting
npm run format

# Type checking
npm run type-check
```

---

## Project Structure

```
modules/modem/
├── backend/
│   ├── main.py                  # FastAPI server + WebSocket
│   ├── config.py                # Configuration (pydantic-settings)
│   ├── requirements.txt         # Python dependencies
│   │
│   ├── api/                     # REST endpoints
│   ├── protocol/                # CASCADE protocol (message format, kernels, nets)
│   ├── model/                   # PyTorch model (decode, encode)
│   ├── hardware/                # Radio (Hamlib), Audio (sounddevice)
│   ├── state/                   # Server state management
│   └── telemetry/              # Telemetry collection (Phase 7)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main app
│   │   ├── contexts/            # React Context (state management)
│   │   ├── hooks/               # Custom hooks (useWebSocket, etc.)
│   │   ├── components/          # React components
│   │   │   ├── ui/              # shadcn/ui components
│   │   │   ├── layout/          # Header, MainLayout
│   │   │   ├── radio/           # Radio control
│   │   │   ├── conversations/   # Message threads
│   │   │   ├── network/         # Topology graphs
│   │   │   └── net_ops/         # Net operations
│   │   └── types/               # TypeScript types
│   │
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── tests/
│   ├── backend/                 # pytest tests
│   └── frontend/                # vitest + Playwright tests
│
├── README.md                    # This file
├── ARCHITECTURE.md              # System design
├── API.md                       # Protocol reference
├── CONTRIBUTING.md              # Developer guide
└── DEPLOYMENT.md                # Production setup
```

---

## Usage

### Basic QSO (Contact)

1. **Start application** (backend + frontend)
2. **Set frequency** (e.g., 14.074 MHz for 20m)
3. **Monitor** - Decoded messages appear in conversation list
4. **Send message**:
   - Click conversation or "New Message"
   - Enter destination callsign (e.g., W2DEF)
   - Type message
   - Click "Send"

### Net Operations

**As Net Controller:**
1. Click "Form Net"
2. Select profile (DX / Ragchew / Emergency / Contest)
3. Enter purpose/description
4. Wait for check-ins
5. Assign slots from pending requests
6. Broadcast schedule

**As Net Member:**
1. See net announcement
2. Click "Join Net"
3. Compose message
4. Request slot (message pre-encoded, exact duration shown)
5. Wait for slot assignment
6. Transmit at assigned time

### Emergency Relay

If you receive emergency relay request:
1. Modal dialog appears with emergency details
2. Review: origin, destination, message, route
3. Click "APPROVE" to relay (or "DENY")
4. Your station relays to next hop

---

## Troubleshooting

### Radio Not Connecting

**Check:**
- Hamlib model number correct? (see `.env`)
- Serial port correct? (`/dev/ttyUSB0` on Linux, `COM3` on Windows)
- CAT cable connected?
- Radio CAT/CI-V enabled in radio menu?
- Baud rate matches radio setting?

**Test:**
```bash
# Linux: Check USB serial devices
ls -l /dev/ttyUSB*

# Test with rigctl (Hamlib command-line tool)
rigctl -m 3037 -r /dev/ttyUSB0 -s 9600
# Commands: f (get frequency), \set_freq 14074000
```

### Audio Issues

**Check:**
- Sound card connected?
- Correct input/output devices selected?
- Audio levels appropriate? (not clipping, not too quiet)

**List audio devices:**
```python
python -c "import sounddevice; print(sounddevice.query_devices())"
```

Set device index in `.env`:
```env
AUDIO_INPUT_DEVICE=2   # Index from query_devices()
AUDIO_OUTPUT_DEVICE=3
```

### WebSocket Connection Failed

**Check:**
- Backend running? (http://localhost:8000/api/health should respond)
- Firewall blocking port 8000?
- CORS issues? (frontend on different port needs CORS enabled)

**Test WebSocket manually:**
```javascript
// Browser console
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onopen = () => console.log('Connected');
ws.onmessage = (e) => console.log('Message:', JSON.parse(e.data));
```

### Frontend Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite
npm run dev
```

---

## Remote Access

To access CASCADE from other devices on your network:

1. **Find your computer's IP:**
   ```bash
   # Linux/Mac
   ip addr show
   # or
   ifconfig

   # Windows
   ipconfig
   ```

2. **Update backend `.env`:**
   ```env
   HOST=0.0.0.0  # Listen on all interfaces
   ```

3. **Update frontend WebSocket URL** (in `src/hooks/useWebSocket.ts`):
   ```typescript
   // Change from:
   const ws = new WebSocket('ws://localhost:8000/ws');

   // To:
   const ws = new WebSocket('ws://YOUR_IP:8000/ws');
   // e.g., ws://192.168.1.100:8000/ws
   ```

4. **Access from other devices:**
   - Desktop/laptop browser: `http://YOUR_IP:8000`
   - Tablet/phone browser: `http://YOUR_IP:8000`

---

## Roadmap

### Phase 1: Foundation ✅ (Current)
- [x] FastAPI server with native WebSockets
- [x] React frontend with shadcn/ui
- [x] Hamlib radio control
- [ ] Audio I/O and spectrum display

### Phase 2: Single-User Communication
- [ ] Binary message format (19 + payload + 8 bytes)
- [ ] CRC32 + xxHash32 validation
- [ ] PyTorch model integration (single user decode)
- [ ] Send/receive messages

### Phase 3: Kernel System
- [ ] RX kernel generation (64-bit)
- [ ] 3-round exchange (prokernel → antikernel → adaptation)
- [ ] Available tone encoding (40-bit run-length)
- [ ] Adaptive encoding (BPSK/QPSK/8-QAM)

### Phase 4: Multi-User Chaos
- [ ] Decode up to 45 simultaneous users
- [ ] Pattern correlation for all 128 patterns
- [ ] Conversation threading
- [ ] Multi-user UI (conversation list)

### Phase 5: Network Operations
- [ ] Net formation (controller, members, relays)
- [ ] Net profiles (DX/Ragchew/Emergency/Contest)
- [ ] Slot scheduling (pre-encoded, exact durations)
- [ ] Network topology graph (D3.js)

### Phase 6: Emergency Relay
- [ ] Emergency message detection
- [ ] Manual relay approval UI
- [ ] Multi-hop tracking
- [ ] Rate limiting (5 msg/hour)

### Phase 7: Production
- [ ] Telemetry collection (neural state)
- [ ] Deployment scripts (systemd, Docker)
- [ ] Complete documentation
- [ ] CI/CD pipeline

---

## Learn More

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and data flow
- **[API.md](API.md)** - REST + WebSocket protocol reference
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Developer guide and coding standards
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment instructions

### CASCADE Protocol

See main CASCADE documentation in `/docs/`:
- `architecture.md` - 128-pattern chaos system overview
- `protocol/` - Message format, kernels, nets, emergency relay
- `model/` - Neural network architecture details

---

## License

See main CASCADE project LICENSE file.

## Contributors

Built as part of the CASCADE (Cognitive Adaptive Spectrum Coordination And Distributed Efficiency) project.

---

**Last updated:** 2025-10-04
**Version:** 0.1.0-alpha
