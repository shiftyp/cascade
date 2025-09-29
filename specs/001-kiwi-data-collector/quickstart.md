# Quickstart: KiwiSDR Data Collector

## Prerequisites

- Python 3.11 or higher
- 15TB available storage
- Internet connection (minimum 10 Mbps)
- Linux (Ubuntu 22.04 recommended) or macOS

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/cascade/cascade.git
cd cascade/modules/data
```

### 2. Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- kiwiclient>=0.1.0
- numpy>=1.24.0
- scipy>=1.10.0
- pandas>=2.0.0
- psycopg2>=2.9.0
- pyyaml>=6.0
- python-flac>=0.3.0

### 4. Setup Database
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb cascade_data

# Run migrations
python scripts/setup_database.py
```

### 5. Configure KiwiSDR Sources
Create `config/kiwisdr_sources.yaml`:
```yaml
sources:
  - url: "sdr1.example.com:8073"
    name: "NA East Coast"
    grid: "FN42"
  - url: "sdr2.example.com:8073"
    name: "EU Central"
    grid: "JO62"
  # Add 20-30 more sources for rotation
```

## First Collection

### 1. Test Single Recording
```bash
python -m cascade_collector test \
  --url "websdr.ewi.utwente.nl:8073" \
  --frequency 14074000 \
  --duration 60 \
  --output test_recording.wav
```

Expected output:
```
Connecting to websdr.ewi.utwente.nl:8073...
Connected! GPS: 52.24N 6.86E
Recording 14074.000 kHz for 60 seconds...
Recording complete: test_recording.wav (10.4 MB)
FT8 signals detected: 23
Average SNR: -12 dB
```

### 2. Start Continuous Collection
```bash
python -m cascade_collector start \
  --config config/collection_schedule.yaml \
  --bands 20m,40m,80m \
  --stations 6 \
  --mode continuous
```

### 3. Monitor Progress
```bash
python -m cascade_collector status
```

Expected output:
```
CASCADE Data Collector Status
=============================
Active Stations: 6/6
Current Rate: 144 hours/day
Total Collected: 1,234 hours

Band Coverage:
  80m: 205 hours (2.1%)
  40m: 412 hours (4.1%)
  20m: 617 hours (6.2%)

Storage:
  Used: 213 GB / 15 TB
  Compression: 52% (FLAC)

Next Rotation: 14:30 UTC
Solar: SFI=156 K=3
```

## Validation

### 1. Check Data Quality
```bash
python -m cascade_validator check \
  --recording test_recording.wav
```

Expected output:
```
✓ Sample rate: 12000 Hz
✓ Bit depth: 16-bit
✓ Channels: 2 (IQ)
✓ GPS timestamps: Present
✓ FT8 signals: 23 decoded
✓ QRN windows: 3 segments > 1 sec
✓ Quality score: 0.89 (GOOD)
```

### 2. Extract Propagation Data
```bash
python -m cascade_processor extract \
  --input test_recording.wav \
  --output propagation.json
```

### 3. Verify Storage
```bash
python -m cascade_storage verify
```

## Troubleshooting

### Connection Issues
```bash
# Test KiwiSDR availability
python scripts/test_kiwisdr.py --url sdr.example.com:8073

# Check network
ping sdr.example.com
```

### Storage Issues
```bash
# Check disk space
df -h /data/cascade

# Compress old recordings
python scripts/compress_archives.py --older-than 7
```

### Processing Bottlenecks
```bash
# Monitor CPU usage
htop

# Adjust parallel workers
export CASCADE_WORKERS=4
```

## Next Steps

1. **Scale Up Collection**
   - Add more KiwiSDR sources
   - Enable all 6 HF bands
   - Increase to 12-20 stations

2. **Automate Operations**
   - Setup systemd service
   - Configure log rotation
   - Enable email alerts

3. **Monitor Quality**
   - Dashboard at http://localhost:8080
   - Grafana metrics
   - Daily reports

## Expected Results

After 24 hours you should have:
- 144 hours of recordings (6 stations × 24 hours)
- ~25 GB of compressed IQ data
- 10,000+ FT8 signals decoded
- 1,000+ unique propagation paths
- Complete diurnal cycle coverage

## Support

- Documentation: `/docs/data-collector/`
- Issues: GitHub Issues
- Logs: `/var/log/cascade/collector.log`

---
*Last updated: 2025-09-29*