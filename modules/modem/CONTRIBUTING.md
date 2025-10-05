# Contributing to CASCADE Modem

Thank you for your interest in contributing to CASCADE! This guide will help you get started.

---

## Development Setup

### Prerequisites

**Backend:**
- Python 3.11 or newer
- pip (Python package manager)
- Optional: Hamlib installed (`libhamlib.so.4` on Linux)

**Frontend:**
- Node.js 20 or newer
- npm (comes with Node.js)

**Tools:**
- Git
- Code editor (VS Code recommended)

---

### Initial Setup

```bash
# Clone repository
git clone https://github.com/yourorg/cascade.git
cd cascade/modules/modem

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dev dependencies

# Frontend setup
cd ../frontend
npm install

# Optional: Install pre-commit hooks
cd ../..
pip install pre-commit
pre-commit install
```

---

## Development Workflow

### 1. Create a Branch

```bash
git checkout develop  # Start from develop branch
git pull origin develop
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/topic` - Documentation only
- `refactor/component-name` - Code refactoring
- `test/test-description` - Adding tests

### 2. Make Changes

**Backend:**
```bash
cd backend
source venv/bin/activate
python main.py  # Run server (mock mode if no hardware)
```

**Frontend:**
```bash
cd frontend
npm run dev  # Vite dev server with hot reload
```

### 3. Run Tests

**Backend:**
```bash
cd backend
pytest                    # All tests
pytest -v                 # Verbose
pytest --cov=.           # With coverage
pytest -k test_name      # Specific test
```

**Frontend:**
```bash
cd frontend
npm test                 # Unit tests
npm run test:watch      # Watch mode
npm run test:coverage   # With coverage
npm run test:e2e        # End-to-end (Playwright)
```

### 4. Code Quality Checks

**Backend:**
```bash
# Linting
ruff check .

# Formatting
black .

# Type checking (if using mypy)
mypy .

# Fix all linting issues automatically
ruff check . --fix
black .
```

**Frontend:**
```bash
# Linting
npm run lint

# Formatting
npm run format

# Type checking
npm run type-check

# Fix all issues
npm run lint:fix
npm run format
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: Add spectrum waterfall display"
```

**Commit Message Format:**

```
<type>: <subject>

<body (optional)>

<footer (optional)>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style (formatting, no logic change)
- `refactor:` Code refactoring (no functional change)
- `test:` Adding/updating tests
- `chore:` Tooling, dependencies, etc.

**Examples:**
```
feat: Add multi-user conversation list component

Implements conversation list that shows up to 45 simultaneous users
with SNR indicators and kernel status.

Closes #123
```

```
fix: WebSocket reconnection on connection loss

Added exponential backoff for reconnection attempts.
```

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create Pull Request on GitHub targeting `develop` branch.

---

## Project Structure

### Backend

```
backend/
├── main.py                  # Entry point
├── config.py                # Configuration
├── api/
│   ├── routes.py           # REST endpoints
│   └── websocket.py        # WebSocket handlers
├── protocol/
│   ├── message_format.py   # Binary serialization
│   ├── validation.py       # CRC32 + xxHash32
│   └── kernel_lifecycle.py # Kernel exchange
├── model/
│   ├── cascade_model.py    # PyTorch model
│   └── multi_decode.py     # 45-user decode
├── hardware/
│   ├── radio_control.py    # Hamlib interface
│   └── audio_io.py         # sounddevice
├── state/
│   └── server_state.py     # Global state
└── tests/
    ├── test_api/
    ├── test_protocol/
    └── test_model/
```

### Frontend

```
frontend/src/
├── App.tsx                  # Main app
├── contexts/
│   └── CASCADEContext.tsx  # Global state (Context + useReducer)
├── hooks/
│   ├── useWebSocket.ts     # WebSocket connection
│   └── useConversations.ts # Conversation logic
├── components/
│   ├── ui/                 # shadcn/ui components
│   ├── layout/             # Header, MainLayout
│   ├── radio/              # Radio controls
│   ├── conversations/      # Message threads
│   ├── network/            # Topology graph
│   └── net_ops/            # Net operations
└── types/
    ├── message.ts          # Message types
    └── network.ts          # Network types
```

---

## Code Style

### Python (Backend)

**Use `black` for formatting:**
```python
# Good: black formatted
def decode_message(signal: np.ndarray, kernel: bytes) -> Optional[Message]:
    """Decode CASCADE message from audio signal

    Args:
        signal: Audio samples (12 kHz)
        kernel: 64-bit RX kernel

    Returns:
        Decoded message or None if validation fails
    """
    patterns = correlate_patterns(signal)
    if not patterns:
        return None

    return reconstruct_message(patterns, kernel)
```

**Use type hints:**
```python
# Good: Type hints
from typing import List, Optional, Dict

def get_active_users(
    conversations: Dict[str, Conversation]
) -> List[str]:
    return [c for c in conversations if c.is_active()]

# Bad: No type hints
def get_active_users(conversations):
    return [c for c in conversations if c.is_active()]
```

**Docstrings (Google style):**
```python
def encode_kernel(
    modulation: str,
    hardware_tier: str,
    available_tones: List[int]
) -> bytes:
    """Encode 64-bit RX kernel

    Args:
        modulation: Preferred modulation (BPSK/QPSK/8-QAM)
        hardware_tier: Hardware capability (rpi4/desktop/gpu)
        available_tones: Which of 78 tones receiver can decode

    Returns:
        64-bit kernel as bytes

    Raises:
        ValueError: If available_tones invalid
    """
    # Implementation...
```

### TypeScript (Frontend)

**Use Prettier for formatting:**
```typescript
// Good: Prettier formatted
interface Message {
  from: string;
  to: string;
  content: string;
  snr: number;
  timestamp: number;
}

function decodeMessage(data: ArrayBuffer): Message | null {
  // Implementation...
}
```

**Use explicit types:**
```typescript
// Good: Explicit types
const [messages, setMessages] = useState<Message[]>([]);

// Bad: Implicit any
const [messages, setMessages] = useState([]);
```

**Use functional components and hooks:**
```typescript
// Good: Functional component with hooks
export function ConversationList() {
  const { state } = useCASCADE();
  const conversations = useConversations();

  return (
    <div>
      {conversations.map(conv => (
        <ConversationThread key={conv.callsign} {...conv} />
      ))}
    </div>
  );
}

// Avoid: Class components (legacy)
class ConversationList extends React.Component { ... }
```

**JSDoc comments for exported functions:**
```typescript
/**
 * Connect to CASCADE WebSocket server
 *
 * @param url - WebSocket server URL
 * @returns WebSocket connection hook
 */
export function useWebSocket(url: string) {
  // Implementation...
}
```

---

## Testing Guidelines

### Backend Tests (pytest)

**Test structure:**
```python
# tests/test_protocol/test_message_format.py

def test_message_serialization():
    """Test binary message serialization"""
    msg = Message(
        from_hash=0x12345678,
        to_hash=0x87654321,
        message_id=1,
        priority=Priority.NORMAL,
        payload="Hello CASCADE"
    )

    serialized = serialize_message(msg)

    assert len(serialized) == 19 + 13 + 8  # Header + payload + validation
    assert serialized[0:4] == b'\x78\x56\x34\x12'  # from_hash (little-endian)
```

**Mock external dependencies:**
```python
@pytest.fixture
def mock_radio():
    """Mock Hamlib radio for testing"""
    with patch('hardware.radio_control.RadioControl') as mock:
        mock.return_value.set_frequency.return_value = None
        yield mock

def test_frequency_change(mock_radio):
    radio = RadioControl(model=None, mock=True)
    radio.set_frequency(14074000)

    mock_radio.return_value.set_frequency.assert_called_once_with(14074000)
```

### Frontend Tests (vitest)

**Component tests:**
```typescript
// tests/components/Header.test.tsx

import { render, screen } from '@testing-library/react';
import { Header } from '@/components/layout/Header';

describe('Header', () => {
  it('displays callsign and frequency', () => {
    render(<Header />);

    expect(screen.getByText('W1ABC')).toBeInTheDocument();
    expect(screen.getByText(/14.074 MHz/)).toBeInTheDocument();
  });
});
```

**Hook tests:**
```typescript
// tests/hooks/useWebSocket.test.ts

import { renderHook } from '@testing-library/react';
import { useWebSocket } from '@/hooks/useWebSocket';

describe('useWebSocket', () => {
  it('connects to server', () => {
    const { result } = renderHook(() =>
      useWebSocket('ws://localhost:8000/ws')
    );

    expect(result.current.connected).toBe(false);
    // Wait for connection...
    expect(result.current.connected).toBe(true);
  });
});
```

### E2E Tests (Playwright)

```typescript
// tests/e2e/radio-control.spec.ts

import { test, expect } from '@playwright/test';

test('change frequency via UI', async ({ page }) => {
  await page.goto('http://localhost:5173');

  // Enter frequency
  await page.fill('input[placeholder*="frequency"]', '14074000');
  await page.click('button:has-text("Set Frequency")');

  // Verify frequency changed
  await expect(page.locator('text=/14.074 MHz/')).toBeVisible();
});
```

---

## Mock Mode Development

### Why Mock Mode?

Develop CASCADE modem **without radio hardware**:
- Backend simulates Hamlib radio control
- Backend simulates audio input/output
- Decoded messages injected for testing
- Frontend works normally

### Using Mock Mode

Mock mode activates automatically if Hamlib not found:

```bash
cd backend
python main.py  # Runs in mock mode (no hardware)

# Mock mode logs:
# [INFO] Hamlib library not loaded
# [INFO] Using mock mode
# [INFO] [MOCK] Connected to radio on /dev/ttyUSB0
# [INFO] [MOCK] Set frequency: 14.074 MHz
```

### Testing with Mock Mode

```python
# Inject mock decoded message
async def inject_test_message():
    """Inject test message (mock decode)"""
    await broadcast({
        'type': 'message_decoded',
        'data': {
            'from': 'W2DEF',
            'to': 'W1ABC',
            'content': 'Test message from mock',
            'snr': 12,
            'timestamp': time.time()
        }
    })

# Call periodically to simulate traffic
```

---

## Pull Request Process

### Before Creating PR

1. ✅ All tests pass
2. ✅ Code linted and formatted
3. ✅ Type checking passes
4. ✅ Add tests for new features
5. ✅ Update documentation if needed
6. ✅ Commit messages follow convention

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Lint/format passes
- [ ] Type checking passes
```

### Review Process

1. Create PR targeting `develop` branch
2. CI runs automated tests
3. Maintainer reviews code
4. Address feedback
5. Maintainer approves and merges

---

## Release Process

1. Create release branch: `release/v0.2.0`
2. Update version numbers:
   - `backend/main.py` (`version = "0.2.0"`)
   - `frontend/package.json` (`"version": "0.2.0"`)
3. Update `CHANGELOG.md`
4. Merge to `main`
5. Tag: `git tag v0.2.0`
6. Merge `main` back to `develop`

---

## Getting Help

- **Issues:** GitHub Issues for bugs/features
- **Discussions:** GitHub Discussions for questions
- **Documentation:** See README, ARCHITECTURE, API docs

---

## Code of Conduct

Be respectful, inclusive, and constructive. See main CASCADE project CODE_OF_CONDUCT.md.

---

**Happy coding! 73 de CASCADE Team**
