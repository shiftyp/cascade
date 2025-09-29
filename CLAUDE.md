# CASCADE KiwiSDR Data Collector - Claude Development Guide

## Project Context
CASCADE is building a neural network-based adaptive radio system. You're working on the Data Module, which collects 150,000-300,000 hours of real-world HF radio data (minimum 10,000 hours per band baseline) from the global KiwiSDR network over 18 months.

## Current Feature: KiwiSDR Data Collector
- **Branch**: 001-kiwi-data-collector
- **Stage**: Implementation starting
- **Goal**: Collect QRN and FT8/WSPR propagation data for model training

## Technical Stack
- **Language**: Python 3.11
- **Key Libraries**: kiwiclient, numpy, scipy, pandas, psycopg2
- **Storage**: 35-75TB Tigris S3 with FLAC compression (45-55% size reduction)
- **Database**: PostgreSQL for metadata

## Module Structure
```
modules/data/
├── src/
│   ├── collectors/     # KiwiSDR connections
│   ├── processors/     # FT8/WSPR decoders
│   ├── validators/     # Quality checks
│   └── storage/        # File management
└── tests/             # Pytest test suite
```

## Key Implementation Details

### KiwiSDR Integration
- Use kiwiclient library for connections
- 12 kHz IQ mode at 12 kHz sample rate
- Rotate through 30+ public SDRs to avoid limits
- Handle 90-minute daily usage limits per SDR

### Data Processing
- Real-time FT8 extraction during collection
- Anonymize callsigns via one-way hashing
- FLAC compression for 45-55% size reduction
- Keep all data regardless of quality

### Collection Strategy
- 6 stations continuous (one per HF band)
- Scale to 20+ stations during solar events
- Target 200-500 hours/day collection rate
- 18-month timeline for 150,000-300,000 total hours

## Constitutional Requirements
1. **Data-First**: This is the first module - no dependencies
2. **Test-Driven**: Write failing tests before implementation
3. **Real Data**: No synthetic data - only real KiwiSDR recordings
4. **Privacy**: Anonymize all callsigns and locations
5. **Reproducible**: Deterministic processing, versioned datasets

## Testing Requirements
```python
# Test structure for TDD
def test_kiwi_connection():
    """Test connection to KiwiSDR"""
    assert False  # Write this first

def test_ft8_extraction():
    """Test FT8 signal extraction"""
    assert False  # Then implement
```

## Common Commands
```bash
# Run tests
pytest modules/data/tests/ -v

# Start collection
python -m cascade_collector start --bands 20m,40m

# Check status
python -m cascade_collector status

# Validate recording
python -m cascade_validator check recording.wav
```

## Current Tasks
See `/specs/001-kiwi-data-collector/tasks.md` for implementation tasks.

## Known Issues & Solutions
- **KiwiSDR timeout**: Implement exponential backoff
- **Storage overflow**: Auto-prune old processed files
- **CPU bottleneck**: Distribute processing across cores

## Recent Changes
- Updated storage requirement to 35-75TB for 150k-300k hours
- Added distributed Fly.io architecture with Redis/KeyDB coordination
- Added comprehensive collection strategy
- Defined data model and API contracts

---
*Last updated: 2025-09-29 by /plan command*