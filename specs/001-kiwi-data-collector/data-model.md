# Data Model: KiwiSDR Data Collector

## Entity Relationship Overview

```
KiwiSDRSource (1) -----> (*) RecordingSession
RecordingSession (1) -----> (*) QRNSample
RecordingSession (1) -----> (*) PropagationRecord
RecordingSession (*) -----> (1) SpaceWeatherData
CollectionSchedule (1) -----> (*) RecordingSession
SpaceWeatherData (1) -----> (*) CollectionSchedule (triggers)
```

## Core Entities

### RecordingSession
Represents a single continuous recording from one KiwiSDR on one frequency.

**Attributes:**
- `session_id`: UUID - Unique identifier
- `kiwisdr_id`: UUID - Reference to KiwiSDRSource
- `start_time`: DateTime - UTC timestamp of recording start
- `end_time`: DateTime - UTC timestamp of recording end
- `center_frequency_hz`: Integer - Center frequency in Hz
- `bandwidth_hz`: Integer - Recording bandwidth (typically 12000)
- `sample_rate`: Integer - Sample rate in Hz
- `iq_file_path`: String - Path to compressed IQ FLAC file
- `file_size_bytes`: BigInteger - Size of stored file
- `gps_latitude`: Float - Anonymized latitude (rounded)
- `gps_longitude`: Float - Anonymized longitude (rounded)
- `solar_flux_index`: Integer - SFI at recording time
- `k_index`: Integer - Geomagnetic K-index
- `signal_count`: Integer - Number of FT8/WSPR signals detected
- `avg_noise_floor_dbm`: Float - Average noise floor
- `quality_score`: Float - 0.0-1.0 quality metric
- `processing_status`: Enum - [pending, processing, completed, failed]
- `created_at`: DateTime - Database entry creation
- `updated_at`: DateTime - Last modification

**Constraints:**
- start_time < end_time
- center_frequency_hz between 10000 and 30000000
- file_size_bytes > 0
- quality_score between 0.0 and 1.0

**State Transitions:**
- pending → processing → completed
- pending → processing → failed

### SDRSource
Registry of available SDR receivers (KiwiSDR and WebSDR) with complete metadata and usage tracking.

**Attributes:**
- `sdr_id`: UUID - Unique identifier
- `sdr_type`: Enum - [KIWISDR, WEBSDR]
- `url`: String - SDR URL (host:port for KiwiSDR, base URL for WebSDR)
- `name`: String - Friendly name from receiver config
- `institution_type`: Enum - [INDIVIDUAL, UNIVERSITY, RESEARCH_INSTITUTE, AMATEUR_CLUB]
- `grid_square`: String - 4-character Maidenhead locator
- `latitude`: Float - Exact latitude (internal use only)
- `longitude`: Float - Exact longitude (internal use only)
- `altitude_m`: Integer - Altitude in meters
- `antenna_type`: String - Antenna description from config
- `daily_limit_minutes`: Integer - Usage limit per day (typically 30-90)
- `session_limit_minutes`: Integer - Max session duration (typically 30)
- `peak_hours_local`: JSON - Restricted hours in receiver local time
- `usage_policy`: Enum - [PUBLIC_LIMITED, RESEARCH_AGREEMENT, COOPERATIVE, RESTRICTED]
- `owner_contact`: String - Contact for research coordination (hashed)
- `research_approved`: Boolean - Approved for extended research usage
- `last_used`: DateTime - Last connection time
- `usage_today_minutes`: Integer - Minutes used today from our IPs
- `total_ips_used`: Integer - Number of IP addresses used today
- `next_available`: DateTime - When daily quota resets
- `reliability_score`: Float - 0.0-1.0 uptime metric
- `is_active`: Boolean - Currently available
- `has_gps`: Boolean - GPS timestamps available
- `sdr_metadata`: JSON - Complete SDR configuration (format varies by type)
  - `version`: KiwiSDR software version
  - `fpga_version`: FPGA firmware version
  - `board_type`: Hardware revision
  - `adc_clk_hz`: ADC clock frequency
  - `if_freq_hz`: IF frequency
  - `max_channels`: Maximum IQ channels
  - `max_users`: Maximum concurrent users
  - `current_users`: Users at discovery
  - `frequency_min_khz`: Minimum tunable frequency
  - `frequency_max_khz`: Maximum tunable frequency
  - `sampling_rate_hz`: Native sampling rate
  - `gps_good`: GPS lock status
  - `gps_fixes`: Number of GPS fixes
  - `uptime_days`: System uptime
  - `cpu_temp_c`: CPU temperature
  - `admin_email`: Contact (hashed)
  - `registration_status`: Public/private
  - `tdoa_enabled`: TDoA participation
  - `dx_labels`: DX label count
- `network_stats`: JSON - Connection quality metrics
  - `avg_latency_ms`: Average ping time
  - `packet_loss_percent`: Measured packet loss
  - `bandwidth_mbps`: Available bandwidth
  - `connection_drops`: Historical drop count
- `created_at`: DateTime - First discovered
- `updated_at`: DateTime - Last checked
- `last_metadata_fetch`: DateTime - Last full metadata update

**Constraints:**
- url unique
- daily_limit_minutes >= 0 (0 = unlimited for some WebSDRs)
- reliability_score between 0.0 and 1.0
- grid_square matches [A-R]{2}[0-9]{2} pattern
- sdr_type in [KIWISDR, WEBSDR]
- institution_type required if sdr_type = WEBSDR

### QRNSample
Characterization of atmospheric noise within a recording.

**Attributes:**
- `sample_id`: UUID - Unique identifier
- `session_id`: UUID - Reference to RecordingSession
- `timestamp`: DateTime - UTC time within recording
- `frequency_hz`: Integer - Frequency of measurement
- `bandwidth_hz`: Integer - Analysis bandwidth
- `noise_floor_dbm`: Float - Measured noise floor
- `peak_amplitude_dbm`: Float - Peak noise spike
- `rms_amplitude_dbm`: Float - RMS noise level
- `impulse_count`: Integer - Number of impulse events
- `occupancy_percent`: Float - Spectral occupancy
- `statistical_params`: JSON - Additional statistics
  - `mean`, `variance`, `skewness`, `kurtosis`
  - `percentiles`: [10, 25, 50, 75, 90]
- `created_at`: DateTime - Analysis timestamp

**Constraints:**
- timestamp within parent session's start/end
- occupancy_percent between 0.0 and 100.0

### PropagationRecord
Extracted FT8/WSPR signal mutations for propagation characterization.

**Attributes:**
- `record_id`: UUID - Unique identifier
- `session_id`: UUID - Reference to RecordingSession
- `timestamp`: DateTime - Signal reception time
- `mode`: Enum - [FT8, WSPR]
- `tx_callsign_hash`: String - Anonymized transmitter ID
- `tx_grid`: String - 4-character grid
- `rx_callsign_hash`: String - Anonymized receiver ID
- `rx_grid`: String - 4-character grid
- `frequency_hz`: Integer - Signal frequency
- `snr_db`: Float - Signal-to-noise ratio
- `drift_hz`: Float - Frequency drift
- `distance_km`: Float - Great circle distance
- `azimuth_deg`: Float - Bearing from RX to TX
- `mutation_data`: JSON - Detailed propagation effects
  - `frequency_spread_hz`: Array of per-symbol values
  - `amplitude_fading`: Array of per-symbol values
  - `phase_rotation`: Array of per-symbol values
  - `multipath_delays_ms`: Detected delays
  - `doppler_spread_hz`: Measured Doppler
- `propagation_mode`: String - Estimated mode (F2, Es, etc.)
- `decoded_successfully`: Boolean - Whether signal decoded
- `created_at`: DateTime - Extraction timestamp

**Constraints:**
- mode in [FT8, WSPR]
- snr_db between -50 and +50
- distance_km >= 0

### SpaceWeatherData
Captures NOAA space weather conditions for correlating with propagation.

**Attributes:**
- `weather_id`: UUID - Unique identifier
- `timestamp`: DateTime - Observation time (UTC)
- `source`: Enum - [NOAA_SWPC, NOAA_API, MANUAL]
- `solar_flux_index`: Integer - 10.7cm solar flux (SFU)
- `k_index`: Integer - Planetary K-index (0-9)
- `ap_index`: Integer - Planetary Ap index
- `sunspot_number`: Integer - Daily sunspot count
- `xray_flux`: Float - X-ray flux (W/m²)
- `xray_class`: String - X-ray flare class (A, B, C, M, X)
- `proton_flux`: Float - Proton flux (pfu)
- `electron_flux`: Float - Electron flux (electrons/cm²-s-sr)
- `magnetometer_data`: JSON - Magnetometer readings
  - `bx_nt`: Interplanetary magnetic field X
  - `by_nt`: Interplanetary magnetic field Y
  - `bz_nt`: Interplanetary magnetic field Z
  - `bt_nt`: Total field strength
- `solar_wind`: JSON - Solar wind parameters
  - `speed_km_s`: Solar wind speed
  - `density_cm3`: Particle density
  - `temperature_k`: Temperature
- `aurora_power`: Integer - Northern hemisphere power (GW)
- `dst_index`: Integer - Disturbance storm time index
- `forecast_data`: JSON - 3-day predictions
  - `k_index_forecast`: Array of predicted K values
  - `solar_flux_forecast`: Array of predicted SFI
  - `storm_probability`: Geomagnetic storm probability
- `alerts`: JSON - Active space weather alerts
  - `alert_type`: Type of alert
  - `severity`: Alert severity level
  - `message`: Alert description
- `solar_cycle_phase`: Enum - Current solar cycle phase [MINIMUM, RISING, MAXIMUM, DECLINING]
- `solar_cycle_number`: Integer - Solar cycle number (currently 25)
- `qbo_index`: Float - Quasi-Biennial Oscillation index (-40 to +40)
- `qbo_phase`: Enum - QBO phase [EASTERLY, WESTERLY, TRANSITION]
- `lunar_phase`: Float - Lunar phase (0.0=new moon, 0.5=full moon, 1.0=new moon)
- `lunar_age_days`: Integer - Days since new moon (0-29)
- `season`: Enum - Astronomical season [WINTER, SPRING, SUMMER, AUTUMN]
- `seasonal_balance_factor`: Float - Collection weighting factor for seasonal balance (0.8-1.3)
- `equinoctial_enhancement`: Boolean - True during equinoctial periods (Mar 15-Apr 15, Sep 15-Oct 15)
- `cycle_metadata`: JSON - Additional natural cycle tracking
  - `solar_rotation_number`: Solar rotation since reference
  - `days_since_solar_min`: Days since solar cycle minimum
  - `geomagnetic_season`: Enhanced/suppressed based on IMF orientation
- `created_at`: DateTime - Database entry time
- `updated_at`: DateTime - Last modification

**Constraints:**
- k_index between 0 and 9
- solar_flux_index > 0
- qbo_index between -40 and 40
- lunar_phase between 0.0 and 1.0
- lunar_age_days between 0 and 29
- seasonal_balance_factor between 0.8 and 1.3
- timestamp unique (one entry per observation time)

### CollectionSchedule
Defines automated recording schedules and priorities.

**Attributes:**
- `schedule_id`: UUID - Unique identifier
- `name`: String - Schedule name
- `frequency_hz`: Integer - Target frequency
- `band_name`: String - Amateur band (80m, 40m, etc.)
- `priority`: Integer - 1 (highest) to 10 (lowest)
- `collection_mode`: Enum - [continuous, sampled, triggered]
- `sample_duration_seconds`: Integer - Recording length
- `sample_interval_seconds`: Integer - Time between recordings
- `min_stations`: Integer - Minimum concurrent stations
- `max_stations`: Integer - Maximum concurrent stations
- `geographic_targets`: JSON - Preferred locations
  - `regions`: Array of grid square prefixes
  - `min_spacing_km`: Minimum station separation
- `trigger_conditions`: JSON - Event triggers
  - `k_index_threshold`: Trigger on geomagnetic activity
  - `solar_flux_threshold`: Trigger on SFI
  - `time_of_day`: Specific UTC hours
- `is_active`: Boolean - Schedule enabled
- `total_hours_collected`: Float - Running total
- `target_hours`: Float - Goal (e.g., 10000)
- `created_at`: DateTime - Schedule creation
- `updated_at`: DateTime - Last modification

**Constraints:**
- priority between 1 and 10
- min_stations <= max_stations
- sample_duration_seconds > 0
- target_hours > total_hours_collected

## Relationships

### Primary Keys
- All entities use UUID primary keys for distributed generation
- No auto-increment to avoid conflicts

### Foreign Keys
- RecordingSession.kiwisdr_id → KiwiSDRSource.kiwisdr_id
- QRNSample.session_id → RecordingSession.session_id
- PropagationRecord.session_id → RecordingSession.session_id

### Indexes
- RecordingSession: (start_time, frequency_hz), (kiwisdr_id, created_at)
- KiwiSDRSource: (url), (is_active, reliability_score)
- QRNSample: (session_id, timestamp)
- PropagationRecord: (session_id, timestamp), (tx_grid, rx_grid)
- CollectionSchedule: (is_active, priority)

## Data Validation Rules

### Recording Quality
- Minimum 10 FT8 signals OR 1 WSPR signal per 10-minute recording
- GPS lock required (no timestamp drift)
- Sample rate consistency (±1%)
- No data gaps > 1 second
- SDR metadata must be captured at recording start

### SDR Provenance
- Every recording MUST link to valid KiwiSDRSource record
- SDR metadata snapshot stored with each recording
- Track SDR configuration changes between recordings
- Flag recordings if SDR config differs from baseline
- Maintain chain of custody for all samples

### Geographic Diversity
- Minimum 2000 km between "diverse" stations
- Maximum 3 stations per grid square prefix (e.g., FN)
- Prefer stations with reliability_score > 0.8
- Balance collection across geographic regions

### Temporal Coverage
- Each band needs 24-hour coverage (all UTC hours)
- Seasonal balance (equal collection per month)
- Solar condition balance (K-index distribution)
- Track SDR availability patterns for scheduling

## Privacy Considerations

### Anonymization Process
1. Callsigns hashed with per-deployment salt
2. Grid squares truncated to 4 characters
3. Exact lat/lon never exposed via API
4. Message content never stored
5. No correlation between sessions possible

### Data Retention
- Raw IQ: Compressed and archived
- Metadata: Retained indefinitely
- PII: Never collected

## NOAA Space Weather Integration

### Data Sources
The system connects to multiple NOAA services for comprehensive space weather data:

1. **NOAA Space Weather Prediction Center (SWPC)**
   - Primary source: `https://services.swpc.noaa.gov/json/`
   - Real-time solar flux: `/json/f107_cm_flux.json`
   - Planetary K-index: `/json/planetary_k_index_1m.json`
   - Solar wind: `/json/rtsw/rtsw_wind_1m.json`
   - Magnetometer: `/json/rtsw/rtsw_mag_1m.json`
   - Update frequency: Every 1-5 minutes

2. **NOAA FTP Archive**
   - Historical data: `ftp://ftp.swpc.noaa.gov/pub/indices/`
   - Daily reports: `/DSD.txt` (Daily Solar Data)
   - Forecasts: `/27DO.txt` (27-day outlook)
   - Update frequency: Daily

3. **NOAA Alerts Service**
   - Space weather alerts: `https://services.swpc.noaa.gov/products/alerts.json`
   - Types: Solar flares, geomagnetic storms, radio blackouts
   - Update frequency: As issued

### Collection Strategy
```python
# Polling schedule for NOAA data
NOAA_SCHEDULE = {
    'realtime': {
        'interval_minutes': 5,
        'endpoints': ['f107_cm_flux', 'planetary_k_index_1m'],
        'retry_on_failure': True
    },
    'daily': {
        'time_utc': '00:30',  # After daily update
        'endpoints': ['DSD.txt', '27DO.txt'],
        'store_history': True
    },
    'alerts': {
        'interval_minutes': 15,
        'endpoints': ['alerts.json'],
        'trigger_collection': True  # Trigger surge collection
    }
}
```

### Event-Triggered Collection
Space weather events automatically trigger increased collection:
- **K-index ≥ 5**: Scale to 12 stations (2× baseline)
- **K-index ≥ 7**: Scale to 20+ stations (storm mode)
- **M-class flare**: Focus on affected bands
- **X-class flare**: Maximum collection all bands
- **Proton event**: Polar path emphasis

### Data Correlation
Each recording links to nearest space weather observation:
```sql
-- Link recordings to space weather
UPDATE recording_sessions r
SET
    solar_flux_index = sw.solar_flux_index,
    k_index = sw.k_index
FROM space_weather_data sw
WHERE sw.timestamp = (
    SELECT MAX(timestamp)
    FROM space_weather_data
    WHERE timestamp <= r.start_time
);
```

### Redundancy & Fallbacks
- Cache last 7 days of space weather data locally
- Use previous values if NOAA unavailable
- Manual entry capability for critical events
- Alternative source: `hamqsl.com` solar data API

## SDR Quality Tracking

### SDR Health Monitoring
Track key metrics for each KiwiSDR to ensure data quality:
- **GPS stability**: Number of GPS fixes, lock duration
- **Frequency accuracy**: Measured vs expected (PPB error)
- **ADC performance**: Clipping events, dynamic range
- **Network reliability**: Latency, packet loss, disconnections
- **Thermal stability**: Temperature variations affecting performance

### SDR Selection Algorithm
Prioritize SDRs based on:
1. **GPS lock status** (required for timing accuracy)
2. **Reliability score** (>0.8 preferred)
3. **Network latency** (<200ms preferred)
4. **Geographic location** (maximize diversity)
5. **Available bandwidth** (unused channels)
6. **Historical performance** (successful recordings)

### Recording-SDR Linkage
Each recording maintains full SDR context:
```sql
-- Example query to trace recording provenance
SELECT
    r.session_id,
    r.start_time,
    r.center_frequency_hz,
    s.name as sdr_name,
    s.grid_square,
    s.antenna_type,
    s.kiwi_metadata->>'version' as kiwi_version,
    s.kiwi_metadata->>'gps_good' as gps_status,
    s.network_stats->>'avg_latency_ms' as latency
FROM recording_sessions r
JOIN kiwisdr_sources s ON r.kiwisdr_id = s.kiwisdr_id
WHERE r.quality_score > 0.9;
```

## Collection Dashboard & Monitoring

### Real-Time Dashboard Queries

#### Overall Collection Status
```sql
-- Main dashboard view: Collection progress
CREATE OR REPLACE VIEW v_collection_status AS
SELECT
    -- Overall progress
    COUNT(DISTINCT r.session_id) as total_sessions,
    SUM(EXTRACT(EPOCH FROM (r.end_time - r.start_time))/3600) as total_hours_collected,
    60000 - SUM(EXTRACT(EPOCH FROM (r.end_time - r.start_time))/3600) as hours_remaining,

    -- Daily rate
    SUM(CASE
        WHEN r.start_time >= NOW() - INTERVAL '24 hours'
        THEN EXTRACT(EPOCH FROM (r.end_time - r.start_time))/3600
        ELSE 0
    END) as last_24h_hours,

    -- Active stations
    COUNT(DISTINCT CASE
        WHEN r.start_time >= NOW() - INTERVAL '1 hour'
        THEN r.kiwisdr_id
    END) as active_stations,

    -- Data quality
    AVG(r.quality_score) as avg_quality,
    SUM(r.file_size_bytes)/1e12 as total_storage_tb,

    -- Estimated completion
    CASE
        WHEN SUM(CASE WHEN r.start_time >= NOW() - INTERVAL '7 days'
                 THEN EXTRACT(EPOCH FROM (r.end_time - r.start_time))/3600 END) > 0
        THEN (60000 - SUM(EXTRACT(EPOCH FROM (r.end_time - r.start_time))/3600)) /
             (SUM(CASE WHEN r.start_time >= NOW() - INTERVAL '7 days'
                  THEN EXTRACT(EPOCH FROM (r.end_time - r.start_time))/3600 END) / 7)
        ELSE NULL
    END as estimated_days_to_complete
FROM recording_sessions r
WHERE r.processing_status = 'completed';
```

#### Per-Band Progress
```sql
-- Band-specific collection progress
CREATE OR REPLACE VIEW v_band_progress AS
SELECT
    cs.band_name,
    cs.frequency_hz/1000 as center_freq_khz,
    cs.target_hours,
    cs.total_hours_collected,
    (cs.total_hours_collected / cs.target_hours * 100) as percent_complete,
    COUNT(DISTINCT r.kiwisdr_id) as unique_sdrs,
    COUNT(DISTINCT DATE(r.start_time)) as days_collected,
    AVG(r.signal_count) as avg_signals_per_session,

    -- Geographic diversity
    COUNT(DISTINCT LEFT(s.grid_square, 2)) as grid_prefixes_covered,

    -- Time coverage (UTC hours covered)
    COUNT(DISTINCT EXTRACT(HOUR FROM r.start_time)) as utc_hours_covered,
    24 - COUNT(DISTINCT EXTRACT(HOUR FROM r.start_time)) as utc_hours_missing

FROM collection_schedules cs
LEFT JOIN recording_sessions r ON r.center_frequency_hz = cs.frequency_hz
LEFT JOIN kiwisdr_sources s ON r.kiwisdr_id = s.kiwisdr_id
GROUP BY cs.band_name, cs.frequency_hz, cs.target_hours, cs.total_hours_collected
ORDER BY cs.frequency_hz;
```

#### Active SDR Status
```sql
-- Current SDR activity and health
CREATE OR REPLACE VIEW v_sdr_status AS
SELECT
    s.name,
    s.grid_square,
    s.antenna_type,
    s.reliability_score,
    s.usage_today_minutes,
    s.daily_limit_minutes - s.usage_today_minutes as minutes_available,

    -- Current activity
    CASE
        WHEN EXISTS (
            SELECT 1 FROM recording_sessions r
            WHERE r.kiwisdr_id = s.kiwisdr_id
            AND r.end_time IS NULL
        ) THEN 'RECORDING'
        WHEN s.last_used > NOW() - INTERVAL '1 hour' THEN 'IDLE'
        WHEN s.is_active THEN 'AVAILABLE'
        ELSE 'OFFLINE'
    END as status,

    -- Recent performance
    (SELECT AVG(quality_score)
     FROM recording_sessions r
     WHERE r.kiwisdr_id = s.kiwisdr_id
     AND r.start_time > NOW() - INTERVAL '24 hours') as recent_quality,

    -- GPS status
    (s.kiwi_metadata->>'gps_good')::boolean as gps_locked,
    (s.network_stats->>'avg_latency_ms')::float as latency_ms,

    -- Last recording
    (SELECT MAX(r.start_time)
     FROM recording_sessions r
     WHERE r.kiwisdr_id = s.kiwisdr_id) as last_recording

FROM kiwisdr_sources s
WHERE s.is_active = true
ORDER BY status, s.reliability_score DESC;
```

#### Space Weather Correlation
```sql
-- Current space weather and collection response
CREATE OR REPLACE VIEW v_space_weather_status AS
SELECT
    sw.timestamp,
    sw.solar_flux_index,
    sw.k_index,
    sw.xray_class,

    -- Collection response
    COUNT(DISTINCT r.kiwisdr_id) as active_stations,

    -- Alert status
    CASE
        WHEN sw.k_index >= 7 THEN 'STORM MODE (20+ stations)'
        WHEN sw.k_index >= 5 THEN 'ENHANCED (12 stations)'
        WHEN sw.xray_class IN ('M', 'X') THEN 'FLARE RESPONSE'
        ELSE 'BASELINE (6 stations)'
    END as collection_mode,

    -- Propagation quality
    AVG(pr.snr_db) as avg_propagation_snr,
    MAX(pr.distance_km) as max_propagation_distance,

    -- Alert summary
    sw.alerts->>'message' as active_alert

FROM space_weather_data sw
LEFT JOIN recording_sessions r ON
    r.start_time BETWEEN sw.timestamp - INTERVAL '30 minutes'
    AND sw.timestamp + INTERVAL '30 minutes'
LEFT JOIN propagation_records pr ON
    pr.timestamp BETWEEN sw.timestamp - INTERVAL '30 minutes'
    AND sw.timestamp + INTERVAL '30 minutes'
WHERE sw.timestamp > NOW() - INTERVAL '3 hours'
GROUP BY sw.timestamp, sw.solar_flux_index, sw.k_index, sw.xray_class, sw.alerts
ORDER BY sw.timestamp DESC
LIMIT 1;
```

### Terminal Dashboard Script
```bash
#!/bin/bash
# cascade_dashboard.sh - Real-time collection monitor

while true; do
    clear
    echo "═══════════════════════════════════════════════════════════════"
    echo " CASCADE Data Collection Dashboard - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "═══════════════════════════════════════════════════════════════"

    # Overall status
    psql -d cascade -t -c "
        SELECT format(
            E'\n📊 COLLECTION PROGRESS\n' ||
            '├─ Total Hours: %s / 60,000 (%.1f%%)\n' ||
            '├─ Last 24h: %.1f hours/day\n' ||
            '├─ Active Stations: %s\n' ||
            '├─ Storage Used: %.2f TB\n' ||
            '└─ ETA: %s days\n',
            TO_CHAR(total_hours_collected, 'FM999,999'),
            (total_hours_collected / 60000 * 100),
            last_24h_hours,
            active_stations,
            total_storage_tb,
            COALESCE(estimated_days_to_complete::text, 'N/A')
        )
        FROM v_collection_status;"

    # Per-band progress
    echo "📻 BAND PROGRESS"
    psql -d cascade -t -c "
        SELECT format(
            '├─ %s: %5.1f%% [%s/%s hrs] %s UTC hrs covered',
            RPAD(band_name, 4),
            percent_complete,
            TO_CHAR(total_hours_collected, 'FM999,999'),
            TO_CHAR(target_hours, 'FM999,999'),
            utc_hours_covered
        )
        FROM v_band_progress
        ORDER BY center_freq_khz;"

    # Space weather
    psql -d cascade -t -c "
        SELECT format(
            E'\n☀️ SPACE WEATHER\n' ||
            '├─ SFI: %s | K-index: %s | X-ray: %s\n' ||
            '└─ Mode: %s',
            solar_flux_index,
            k_index,
            COALESCE(xray_class, 'A'),
            collection_mode
        )
        FROM v_space_weather_status
        LIMIT 1;"

    # Active SDRs
    echo -e "\n🔌 ACTIVE SDRS (Top 5)"
    psql -d cascade -t -c "
        SELECT format(
            '├─ %s [%s]: %s | GPS:%s | Q:%.2f',
            RPAD(LEFT(name, 20), 20),
            grid_square,
            RPAD(status, 9),
            CASE WHEN gps_locked THEN '✓' ELSE '✗' END,
            COALESCE(recent_quality, 0)
        )
        FROM v_sdr_status
        WHERE status IN ('RECORDING', 'IDLE')
        ORDER BY status, recent_quality DESC NULLS LAST
        LIMIT 5;"

    # Recent issues
    echo -e "\n⚠️  RECENT ISSUES"
    psql -d cascade -t -c "
        SELECT format(
            '├─ %s: %s',
            TO_CHAR(created_at, 'HH24:MI'),
            LEFT(error_message, 50)
        )
        FROM (
            SELECT created_at,
                   'SDR ' || (SELECT name FROM kiwisdr_sources WHERE kiwisdr_id = r.kiwisdr_id) ||
                   ' - Failed recording' as error_message
            FROM recording_sessions r
            WHERE processing_status = 'failed'
            AND created_at > NOW() - INTERVAL '1 hour'
            ORDER BY created_at DESC
            LIMIT 3
        ) recent_errors;"

    echo "═══════════════════════════════════════════════════════════════"
    echo "Refreshing in 30 seconds... (Ctrl+C to exit)"
    sleep 30
done
```

### Python Dashboard Option
```python
# cascade_monitor.py - Rich terminal dashboard
import psycopg2
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
import time

def get_dashboard_data(conn):
    """Fetch all dashboard data from database"""
    with conn.cursor() as cur:
        # Get overall status
        cur.execute("SELECT * FROM v_collection_status")
        overall = cur.fetchone()

        # Get band progress
        cur.execute("SELECT * FROM v_band_progress ORDER BY center_freq_khz")
        bands = cur.fetchall()

        # Get SDR status
        cur.execute("SELECT * FROM v_sdr_status WHERE status != 'OFFLINE' LIMIT 10")
        sdrs = cur.fetchall()

        # Get space weather
        cur.execute("SELECT * FROM v_space_weather_status LIMIT 1")
        weather = cur.fetchone()

    return overall, bands, sdrs, weather

def create_dashboard():
    """Create rich dashboard layout"""
    # Implementation with Rich library
    pass

if __name__ == "__main__":
    conn = psycopg2.connect("dbname=cascade")
    console = Console()

    with Live(create_dashboard(), refresh_per_second=0.5) as live:
        while True:
            data = get_dashboard_data(conn)
            live.update(create_dashboard(*data))
            time.sleep(30)
```

### Alert Triggers
```sql
-- Create alerts for dashboard
CREATE TABLE collection_alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50),
    severity VARCHAR(20),
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    acknowledged BOOLEAN DEFAULT FALSE,
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_sent_at TIMESTAMP
);

-- Trigger for low collection rate
CREATE OR REPLACE FUNCTION check_collection_rate() RETURNS trigger AS $$
BEGIN
    IF NEW.last_24h_hours < 100 THEN
        INSERT INTO collection_alerts (alert_type, severity, message)
        VALUES ('LOW_RATE', 'WARNING',
                format('Collection rate dropped to %.1f hours/day', NEW.last_24h_hours));
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for SDR failures
CREATE OR REPLACE FUNCTION check_sdr_health() RETURNS trigger AS $$
BEGIN
    IF NEW.processing_status = 'failed' THEN
        INSERT INTO collection_alerts (alert_type, severity, message)
        VALUES ('SDR_FAILURE', 'ERROR',
                format('Recording failed for session %s', NEW.session_id));
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

## Gmail Notification System

### Configuration Tables
```sql
-- Gmail notification configuration
CREATE TABLE notification_config (
    config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    smtp_server VARCHAR(100) DEFAULT 'smtp.gmail.com',
    smtp_port INTEGER DEFAULT 587,
    from_email VARCHAR(255) NOT NULL,
    from_password VARCHAR(255) NOT NULL, -- App-specific password
    to_emails TEXT[] NOT NULL,  -- Array of recipient emails
    cc_emails TEXT[],           -- Optional CC recipients

    -- Notification preferences
    enabled BOOLEAN DEFAULT TRUE,
    min_severity VARCHAR(20) DEFAULT 'WARNING', -- INFO, WARNING, ERROR, CRITICAL

    -- Rate limiting
    max_emails_per_hour INTEGER DEFAULT 10,
    emails_sent_this_hour INTEGER DEFAULT 0,
    hour_counter_reset TIMESTAMP DEFAULT NOW(),

    -- Quiet hours (optional)
    quiet_hours_enabled BOOLEAN DEFAULT FALSE,
    quiet_hours_start TIME,     -- e.g., '22:00'
    quiet_hours_end TIME,       -- e.g., '07:00'
    quiet_hours_timezone VARCHAR(50) DEFAULT 'UTC',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Notification templates
CREATE TABLE notification_templates (
    template_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) UNIQUE NOT NULL,
    subject_template TEXT NOT NULL,
    body_template TEXT NOT NULL,
    include_dashboard_link BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Notification log
CREATE TABLE notification_log (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id UUID REFERENCES collection_alerts(alert_id),
    sent_to TEXT[],
    subject TEXT,
    body TEXT,
    sent_at TIMESTAMP DEFAULT NOW(),
    success BOOLEAN,
    error_message TEXT
);

-- Insert default templates
INSERT INTO notification_templates (alert_type, subject_template, body_template) VALUES
('LOW_RATE',
 '[CASCADE] ⚠️ Collection Rate Alert - {rate} hrs/day',
 'The data collection rate has dropped below target:

Current Rate: {rate} hours/day
Target Rate: 110 hours/day
Active Stations: {stations}

This may delay completion of the 60,000 hour goal.
Check dashboard for details: {dashboard_url}'),

('SDR_FAILURE',
 '[CASCADE] 🔴 SDR Recording Failed - {sdr_name}',
 'Recording failure detected:

SDR: {sdr_name} ({grid_square})
Band: {band}
Time: {timestamp}
Error: {error_message}

Automatic retry scheduled.'),

('STORM_MODE',
 '[CASCADE] ⚡ Space Weather Alert - K={k_index}',
 'Geomagnetic storm detected:

K-Index: {k_index}
Solar Flux: {sfi}
Collection Mode: {mode}
Active Stations: {stations}

System automatically scaled collection to capture propagation enhancement.'),

('STORAGE_WARNING',
 '[CASCADE] 💾 Storage Alert - {percent}% Used',
 'Storage space running low:

Used: {used_tb} TB
Total: {total_tb} TB
Percent Full: {percent}%
Days Remaining: {days_left}

Consider archiving processed data or expanding storage.'),

('MILESTONE',
 '[CASCADE] 🎉 Collection Milestone - {hours} Hours!',
 'Milestone reached:

Total Hours Collected: {hours}
Percent Complete: {percent}%
Estimated Completion: {eta}

Band Progress:
{band_summary}');
```

### Python Gmail Notifier
```python
# gmail_notifier.py - Send alerts via Gmail
import smtplib
import psycopg2
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import time
import logging

class GmailNotifier:
    def __init__(self, db_conn):
        self.conn = db_conn
        self.config = self.load_config()
        self.logger = logging.getLogger(__name__)

    def load_config(self):
        """Load Gmail configuration from database"""
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM notification_config WHERE enabled = true LIMIT 1")
            return cur.fetchone()

    def should_send_email(self):
        """Check rate limits and quiet hours"""
        if not self.config['enabled']:
            return False

        # Check rate limiting
        if self.config['emails_sent_this_hour'] >= self.config['max_emails_per_hour']:
            return False

        # Check quiet hours
        if self.config['quiet_hours_enabled']:
            now = datetime.now()
            start = self.config['quiet_hours_start']
            end = self.config['quiet_hours_end']

            if start <= end:
                if start <= now.time() <= end:
                    return False
            else:  # Quiet hours span midnight
                if now.time() >= start or now.time() <= end:
                    return False

        return True

    def send_email(self, alert):
        """Send email for a specific alert"""
        if not self.should_send_email():
            return False

        # Get template
        template = self.get_template(alert['alert_type'])
        if not template:
            return False

        # Format subject and body
        subject = self.format_template(template['subject_template'], alert)
        body = self.format_template(template['body_template'], alert)

        # Create message
        msg = MIMEMultipart()
        msg['From'] = self.config['from_email']
        msg['To'] = ', '.join(self.config['to_emails'])
        if self.config['cc_emails']:
            msg['Cc'] = ', '.join(self.config['cc_emails'])
        msg['Subject'] = subject

        # Add body
        msg.attach(MIMEText(body, 'plain'))

        # Send email
        try:
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['from_email'], self.config['from_password'])

                recipients = self.config['to_emails'] + (self.config['cc_emails'] or [])
                server.send_message(msg, to_addrs=recipients)

            self.log_notification(alert, subject, body, True)
            self.update_rate_limit()
            return True

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            self.log_notification(alert, subject, body, False, str(e))
            return False

    def get_template(self, alert_type):
        """Fetch notification template"""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM notification_templates WHERE alert_type = %s",
                (alert_type,)
            )
            return cur.fetchone()

    def format_template(self, template, alert):
        """Replace template variables with actual values"""
        # Get additional context data
        context = self.get_alert_context(alert)

        # Basic replacements
        formatted = template
        for key, value in context.items():
            formatted = formatted.replace(f'{{{key}}}', str(value))

        return formatted

    def get_alert_context(self, alert):
        """Get context data for template formatting"""
        context = {}

        # Get current collection status
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM v_collection_status")
            status = cur.fetchone()
            if status:
                context['rate'] = f"{status['last_24h_hours']:.1f}"
                context['stations'] = status['active_stations']
                context['total_hours'] = f"{status['total_hours_collected']:.0f}"

            # Get space weather if relevant
            cur.execute("SELECT * FROM v_space_weather_status LIMIT 1")
            weather = cur.fetchone()
            if weather:
                context['k_index'] = weather['k_index']
                context['sfi'] = weather['solar_flux_index']
                context['mode'] = weather['collection_mode']

        context['dashboard_url'] = 'http://your-server:8080/dashboard'
        context['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

        return context

    def log_notification(self, alert, subject, body, success, error=None):
        """Log notification attempt"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO notification_log
                (alert_id, sent_to, subject, body, success, error_message)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                alert['alert_id'],
                self.config['to_emails'],
                subject,
                body,
                success,
                error
            ))

            # Mark alert as notified
            if success:
                cur.execute("""
                    UPDATE collection_alerts
                    SET notification_sent = true, notification_sent_at = NOW()
                    WHERE alert_id = %s
                """, (alert['alert_id'],))

            self.conn.commit()

    def update_rate_limit(self):
        """Update email rate limit counter"""
        with self.conn.cursor() as cur:
            # Reset counter if hour has passed
            cur.execute("""
                UPDATE notification_config
                SET emails_sent_this_hour =
                    CASE
                        WHEN NOW() - hour_counter_reset > INTERVAL '1 hour'
                        THEN 1
                        ELSE emails_sent_this_hour + 1
                    END,
                    hour_counter_reset =
                    CASE
                        WHEN NOW() - hour_counter_reset > INTERVAL '1 hour'
                        THEN NOW()
                        ELSE hour_counter_reset
                    END
                WHERE config_id = %s
            """, (self.config['config_id'],))
            self.conn.commit()

    def check_and_notify(self):
        """Main loop to check for alerts and send notifications"""
        with self.conn.cursor() as cur:
            # Find unsent alerts matching severity threshold
            cur.execute("""
                SELECT * FROM collection_alerts
                WHERE notification_sent = false
                AND severity >= %s
                AND created_at > NOW() - INTERVAL '24 hours'
                ORDER BY
                    CASE severity
                        WHEN 'CRITICAL' THEN 1
                        WHEN 'ERROR' THEN 2
                        WHEN 'WARNING' THEN 3
                        WHEN 'INFO' THEN 4
                    END,
                    created_at DESC
                LIMIT 5
            """, (self.config['min_severity'],))

            alerts = cur.fetchall()

            for alert in alerts:
                if self.send_email(alert):
                    self.logger.info(f"Sent notification for alert {alert['alert_id']}")
                    time.sleep(2)  # Avoid rapid-fire emails
                else:
                    self.logger.warning(f"Failed to send notification for {alert['alert_id']}")

# Main monitoring loop
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Database connection
    conn = psycopg2.connect("dbname=cascade user=cascade")

    # Initialize notifier
    notifier = GmailNotifier(conn)

    # Check every 5 minutes
    while True:
        try:
            notifier.check_and_notify()
        except Exception as e:
            logging.error(f"Notification check failed: {e}")

        time.sleep(300)  # 5 minutes
```

### Setup Instructions
```bash
# setup_gmail_notifications.sh

# 1. Install dependencies
pip install psycopg2-binary

# 2. Configure Gmail (one-time setup)
echo "
-- Insert your Gmail configuration
INSERT INTO notification_config (
    from_email,
    from_password,  -- Use app-specific password!
    to_emails,
    enabled
) VALUES (
    'your-cascade-email@gmail.com',
    'your-16-char-app-password',  -- Generate at myaccount.google.com/apppasswords
    ARRAY['your-personal@email.com'],
    true
);
" | psql -d cascade

# 3. Run notifier as service
cat > /etc/systemd/system/cascade-notifier.service << EOF
[Unit]
Description=CASCADE Gmail Notifier
After=postgresql.service

[Service]
Type=simple
User=cascade
WorkingDirectory=/opt/cascade
ExecStart=/usr/bin/python3 /opt/cascade/gmail_notifier.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl enable cascade-notifier
systemctl start cascade-notifier
```

## Migration Strategy

### Initial Schema
```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE kiwisdr_sources (...);
CREATE TABLE recording_sessions (...);
CREATE TABLE qrn_samples (...);
CREATE TABLE propagation_records (...);
CREATE TABLE collection_schedules (...);
```

### Future Migrations
- Version migrations via Alembic (Python)
- Forward-only migrations
- Backup before migration
- Rollback plan required

---
*Data model v1.0.0 - 2025-09-29*