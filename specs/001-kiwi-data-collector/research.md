# Research: KiwiSDR Data Collector

## Executive Summary
This research document consolidates technical decisions for collecting 60,000 hours of HF radio data (10,000 hours per band) from the global KiwiSDR network over 18 months.

## 1. KiwiSDR Network Analysis

### Public Receiver Availability
- **Total public KiwiSDRs**: 100-150 active globally
- **Geographic distribution**: Good coverage in NA/EU, sparse in AF/SA
- **Typical availability**: 70-80% uptime per receiver
- **Concurrent users**: 4-8 slots per KiwiSDR

**Decision**: Target 30-40 reliable KiwiSDRs for rotation
**Rationale**: Provides redundancy while respecting usage limits
**Alternatives considered**: WebSDR (rejected - no native API)

### Usage Limits and Rotation Strategy
- **Per-SDR limits**: 30-90 minutes/day typical
- **Session timeouts**: 30 minutes common
- **Exemptions**: Some SDRs allow unlimited with permission

**Decision**: Implement round-robin rotation across 30+ SDRs
**Rationale**: Distributes load, avoids hitting daily limits
**Alternatives considered**: VPN rotation (rejected - unethical)

### API Capabilities
- **kiwiclient**: Python library for programmatic access
- **Data formats**: IQ (2-channel), audio (mono)
- **Sample rates**: 12 kHz (4/8 channel mode), 20.25 kHz (3 channel)
- **GPS timestamps**: Available from most KiwiSDRs

**Decision**: Use kiwiclient with 12 kHz IQ mode
**Rationale**: Native Python, GPS timestamps, adequate bandwidth
**Alternatives considered**: Direct WebSocket (rejected - complex)

## 2. FT8/WSPR Processing

### Optimal Recording Windows
- **FT8 bandwidth**: 50 Hz per signal, 2.8 kHz total activity
- **WSPR bandwidth**: 6 Hz per signal, 200 Hz total
- **Quiet zones needed**: 1-2 kHz adjacent spectrum for QRN

**Decision**: 12 kHz windows centered on FT8 frequencies
**Rationale**: Captures FT8, nearby WSPR, and quiet zones
**Alternatives considered**: Narrow 2.8 kHz (rejected - misses QRN)

### Propagation Mutation Extraction
- **Symbol-level analysis**: 79 symbols per FT8 transmission
- **Mutation types**: Frequency drift, amplitude fading, multipath
- **Processing time**: ~100ms per FT8 decode

**Decision**: Real-time extraction during collection
**Rationale**: Reduces storage, enables smart retention
**Alternatives considered**: Batch processing (rejected - 10x storage)

### Processing Architecture
- **Parallel decoding**: Process multiple bands simultaneously
- **CPU requirements**: ~1 core per 2 bands for real-time
- **Memory requirements**: ~500MB per decoder instance

**Decision**: Distributed processing across multiple cores
**Rationale**: Scales with collection rate
**Alternatives considered**: GPU processing (rejected - overkill)

## 3. Storage Optimization

### Compression Analysis
- **WAV baseline**: 173 MB/hour uncompressed
- **FLAC compression**: 45-55% reduction for IQ data
- **Wavpack**: Similar compression, faster decode
- **GZIP**: Only 20-30% reduction

**Decision**: FLAC for long-term storage
**Rationale**: Best compression ratio, lossless, widely supported
**Alternatives considered**: Wavpack (close second), lossy (rejected)

### Storage Architecture
- **Hot tier (SSD)**: Current day's recordings (200 GB)
- **Warm tier (HDD)**: Recent 30 days (2 TB)
- **Cold tier**: Compressed archive (10-15 TB)

**Decision**: Three-tier storage with automated migration
**Rationale**: Balances performance and cost
**Alternatives considered**: All-SSD (too expensive), all-cloud (latency)

### Cloud vs Local Tradeoffs
- **Local NAS**: $800 for 16TB, no recurring costs
- **AWS S3**: $150/month for 15TB standard storage
- **Glacier**: $15/month but slow retrieval

**Decision**: Local NAS primary, cloud backup for critical data
**Rationale**: Cost-effective for 18-month project
**Alternatives considered**: Pure cloud (rejected - expensive)

## 4. Geographic and Temporal Coverage

### Station Selection Criteria
- **Latitude diversity**: Equatorial, mid-latitude, polar
- **Longitude spread**: 2000-4000 km spacing
- **Reliability**: >80% uptime history
- **Noise floor**: <-100 dBm typical

**Decision**: 8 core stations + 12 surge stations
**Rationale**: Ensures coverage during all propagation conditions
**Alternatives considered**: Random selection (rejected - gaps)

### Temporal Sampling Strategy
- **Continuous**: 6 stations 24/7 (one per band)
- **Sampled**: Additional stations 10 min/hour
- **Surge**: All stations during solar events

**Decision**: Hybrid continuous/sampled approach
**Rationale**: Balances completeness with resource limits
**Alternatives considered**: All continuous (rejected - exceeds limits)

## 5. Data Quality and Validation

### Quality Metrics
- **Signal density**: FT8 signals per minute
- **SNR distribution**: Range of signal strengths
- **Quiet windows**: Continuous periods without signals
- **Geographic diversity**: Unique grid squares heard

**Decision**: Keep all data, tag with quality scores
**Rationale**: Even "poor" data valuable for training
**Alternatives considered**: Aggressive filtering (rejected - data loss)

### Validation Pipeline
- **Real-time checks**: GPS lock, sample rate, data corruption
- **Post-processing**: FT8 decode success, QRN statistics
- **Manual review**: Spot checks of anomalies

**Decision**: Automated validation with exception reporting
**Rationale**: Scales to 60,000 hours
**Alternatives considered**: Manual review (rejected - impractical)

## 6. Privacy and Anonymization

### PII Handling
- **Callsigns**: Hash to consistent anonymous IDs
- **Grid squares**: Truncate to 4-character (100km resolution)
- **Message content**: Never stored
- **IP addresses**: Not logged

**Decision**: One-way hashing with salt
**Rationale**: Preserves path analysis while protecting privacy
**Alternatives considered**: Full removal (rejected - loses paths)

### Legal Compliance
- **Amateur radio**: Public transmissions, no expectation of privacy
- **GDPR**: Anonymization satisfies requirements
- **Data retention**: No PII retained

**Decision**: Aggressive anonymization at collection time
**Rationale**: Prevents downstream privacy issues
**Alternatives considered**: Post-hoc anonymization (rejected - risky)

## Conclusions and Recommendations

### Key Technical Decisions
1. **Collection**: kiwiclient Python library with 12 kHz IQ
2. **Processing**: Real-time FT8/WSPR extraction
3. **Storage**: FLAC compression, 10-15 TB local NAS
4. **Coverage**: 6-20 stations variable by conditions
5. **Privacy**: Immediate anonymization via hashing

### Implementation Priorities
1. KiwiSDR connection manager with rotation
2. Real-time FT8 decoder integration
3. FLAC compression pipeline
4. Metadata database design
5. Collection scheduler

### Risk Mitigation
- **KiwiSDR availability**: Monitor and adapt station list
- **Storage overflow**: Implement automatic pruning
- **Processing bottlenecks**: Scale horizontally
- **Network issues**: Local buffering and retry

### Success Metrics
- Achieve 110 hours/day collection rate
- Maintain >80% uptime
- <5% data loss rate
- Complete geographic coverage
- 60,000 hours in 18 months

---
*Research completed: 2025-09-29*