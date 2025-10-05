-- CASCADE KiwiSDR Data Collector Database Schema
-- Version: 001_initial_schema
-- Date: 2025-09-29

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- SDRSource table (combined KiwiSDR and WebSDR)
CREATE TABLE IF NOT EXISTS sdr_sources (
    sdr_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sdr_type VARCHAR(20) NOT NULL CHECK (sdr_type IN ('KIWISDR', 'WEBSDR')),
    url VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    institution_type VARCHAR(50) CHECK (institution_type IN ('INDIVIDUAL', 'UNIVERSITY', 'RESEARCH_INSTITUTE', 'AMATEUR_CLUB')),
    grid_square VARCHAR(4) CHECK (grid_square ~ '^[A-R]{2}[0-9]{2}$'),
    latitude FLOAT,
    longitude FLOAT,
    altitude_m INTEGER,
    antenna_type VARCHAR(255),
    daily_limit_minutes INTEGER DEFAULT 90,
    session_limit_minutes INTEGER DEFAULT 30,
    peak_hours_local JSONB,
    usage_policy VARCHAR(50) DEFAULT 'PUBLIC_LIMITED',
    owner_contact VARCHAR(255),
    research_approved BOOLEAN DEFAULT FALSE,
    last_used TIMESTAMP WITH TIME ZONE,
    usage_today_minutes INTEGER DEFAULT 0,
    total_ips_used INTEGER DEFAULT 0,
    next_available TIMESTAMP WITH TIME ZONE,
    reliability_score FLOAT DEFAULT 0.5 CHECK (reliability_score >= 0 AND reliability_score <= 1),
    is_active BOOLEAN DEFAULT TRUE,
    has_gps BOOLEAN DEFAULT FALSE,
    sdr_metadata JSONB,
    network_stats JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_metadata_fetch TIMESTAMP WITH TIME ZONE
);

-- RecordingSession table
CREATE TABLE IF NOT EXISTS recording_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sdr_id UUID REFERENCES sdr_sources(sdr_id) ON DELETE CASCADE,
    correlation_id UUID,  -- For paired recordings
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    center_frequency_hz INTEGER NOT NULL CHECK (center_frequency_hz > 10000 AND center_frequency_hz < 30000000),
    bandwidth_hz INTEGER NOT NULL DEFAULT 12000,
    sample_rate INTEGER NOT NULL DEFAULT 12000,
    iq_file_path VARCHAR(1024),
    file_size_bytes BIGINT CHECK (file_size_bytes > 0),
    gps_latitude FLOAT,
    gps_longitude FLOAT,
    solar_flux_index INTEGER,
    k_index INTEGER CHECK (k_index >= 0 AND k_index <= 9),
    signal_count INTEGER DEFAULT 0,
    avg_noise_floor_dbm FLOAT,
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    processing_status VARCHAR(20) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_time_range CHECK (start_time < end_time OR end_time IS NULL)
);

-- QRNSample table
CREATE TABLE IF NOT EXISTS qrn_samples (
    sample_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES recording_sessions(session_id) ON DELETE CASCADE,
    correlation_id UUID,  -- For correlated samples
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    frequency_hz INTEGER NOT NULL,
    bandwidth_hz INTEGER NOT NULL,
    noise_floor_dbm FLOAT,
    peak_amplitude_dbm FLOAT,
    rms_amplitude_dbm FLOAT,
    impulse_count INTEGER DEFAULT 0,
    occupancy_percent FLOAT CHECK (occupancy_percent >= 0 AND occupancy_percent <= 100),
    quiet_periods JSONB,  -- Array of quiet period timestamps
    statistical_params JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- PropagationRecord table
CREATE TABLE IF NOT EXISTS propagation_records (
    record_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES recording_sessions(session_id) ON DELETE CASCADE,
    correlation_id UUID,  -- For correlated propagation
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    mode VARCHAR(10) CHECK (mode IN ('FT8', 'WSPR')),
    tx_callsign_hash VARCHAR(64),
    tx_grid VARCHAR(4),
    rx_callsign_hash VARCHAR(64),
    rx_grid VARCHAR(4),
    frequency_hz INTEGER NOT NULL,
    snr_db FLOAT CHECK (snr_db >= -50 AND snr_db <= 50),
    drift_hz FLOAT,
    distance_km FLOAT CHECK (distance_km >= 0),
    azimuth_deg FLOAT,
    mutation_data JSONB,
    propagation_mode VARCHAR(20),  -- F2, Es, etc.
    decoded_successfully BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- SpaceWeatherData table
CREATE TABLE IF NOT EXISTS space_weather_data (
    weather_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL UNIQUE,
    source VARCHAR(20) DEFAULT 'NOAA_SWPC' CHECK (source IN ('NOAA_SWPC', 'NOAA_API', 'MANUAL')),
    solar_flux_index INTEGER CHECK (solar_flux_index > 0),
    k_index INTEGER CHECK (k_index >= 0 AND k_index <= 9),
    ap_index INTEGER,
    sunspot_number INTEGER,
    xray_flux FLOAT,
    xray_class VARCHAR(10),  -- A, B, C, M, X
    proton_flux FLOAT,
    electron_flux FLOAT,
    magnetometer_data JSONB,
    solar_wind JSONB,
    aurora_power INTEGER,
    dst_index INTEGER,
    forecast_data JSONB,
    alerts JSONB,
    solar_cycle_phase VARCHAR(20) CHECK (solar_cycle_phase IN ('MINIMUM', 'RISING', 'MAXIMUM', 'DECLINING')),
    solar_cycle_number INTEGER DEFAULT 25,
    qbo_index FLOAT CHECK (qbo_index >= -40 AND qbo_index <= 40),
    qbo_phase VARCHAR(20) CHECK (qbo_phase IN ('EASTERLY', 'WESTERLY', 'TRANSITION')),
    lunar_phase FLOAT CHECK (lunar_phase >= 0 AND lunar_phase <= 1),
    lunar_age_days INTEGER CHECK (lunar_age_days >= 0 AND lunar_age_days <= 29),
    season VARCHAR(20) CHECK (season IN ('WINTER', 'SPRING', 'SUMMER', 'AUTUMN')),
    seasonal_balance_factor FLOAT CHECK (seasonal_balance_factor >= 0.8 AND seasonal_balance_factor <= 1.3),
    equinoctial_enhancement BOOLEAN DEFAULT FALSE,
    cycle_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- CollectionSchedule table
CREATE TABLE IF NOT EXISTS collection_schedules (
    schedule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    frequency_hz INTEGER NOT NULL,
    band_name VARCHAR(20),
    priority INTEGER CHECK (priority >= 1 AND priority <= 10),
    collection_mode VARCHAR(20) DEFAULT 'continuous' CHECK (collection_mode IN ('continuous', 'sampled', 'triggered')),
    sample_duration_seconds INTEGER CHECK (sample_duration_seconds > 0),
    sample_interval_seconds INTEGER,
    min_stations INTEGER DEFAULT 1,
    max_stations INTEGER DEFAULT 6,
    geographic_targets JSONB,
    trigger_conditions JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    total_hours_collected FLOAT DEFAULT 0,
    target_hours FLOAT DEFAULT 10000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT valid_station_range CHECK (min_stations <= max_stations)
);

-- NotificationConfig table
CREATE TABLE IF NOT EXISTS notification_config (
    config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    smtp_server VARCHAR(100) DEFAULT 'smtp.gmail.com',
    smtp_port INTEGER DEFAULT 587,
    from_email VARCHAR(255) NOT NULL,
    from_password VARCHAR(255) NOT NULL,
    to_emails TEXT[] NOT NULL,
    cc_emails TEXT[],
    enabled BOOLEAN DEFAULT TRUE,
    min_severity VARCHAR(20) DEFAULT 'WARNING',
    max_emails_per_hour INTEGER DEFAULT 10,
    emails_sent_this_hour INTEGER DEFAULT 0,
    hour_counter_reset TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    quiet_hours_enabled BOOLEAN DEFAULT FALSE,
    quiet_hours_start TIME,
    quiet_hours_end TIME,
    quiet_hours_timezone VARCHAR(50) DEFAULT 'UTC',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- CollectionAlerts table
CREATE TABLE IF NOT EXISTS collection_alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    message TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged BOOLEAN DEFAULT FALSE,
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_sent_at TIMESTAMP WITH TIME ZONE
);

-- NotificationTemplates table
CREATE TABLE IF NOT EXISTS notification_templates (
    template_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) UNIQUE NOT NULL,
    subject_template TEXT NOT NULL,
    body_template TEXT NOT NULL,
    include_dashboard_link BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_recording_sessions_start_time ON recording_sessions(start_time);
CREATE INDEX idx_recording_sessions_frequency ON recording_sessions(center_frequency_hz);
CREATE INDEX idx_recording_sessions_sdr_id ON recording_sessions(sdr_id, created_at);
CREATE INDEX idx_sdr_sources_active ON sdr_sources(is_active, reliability_score);
CREATE INDEX idx_sdr_sources_url ON sdr_sources(url);
CREATE INDEX idx_qrn_samples_session ON qrn_samples(session_id, timestamp);
CREATE INDEX idx_propagation_records_session ON propagation_records(session_id, timestamp);
CREATE INDEX idx_propagation_records_grids ON propagation_records(tx_grid, rx_grid);
CREATE INDEX idx_collection_schedules_active ON collection_schedules(is_active, priority);
CREATE INDEX idx_space_weather_timestamp ON space_weather_data(timestamp);
CREATE INDEX idx_collection_alerts_unsent ON collection_alerts(notification_sent, created_at) WHERE notification_sent = FALSE;

-- Update triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_sdr_sources_updated_at BEFORE UPDATE ON sdr_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_recording_sessions_updated_at BEFORE UPDATE ON recording_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_space_weather_data_updated_at BEFORE UPDATE ON space_weather_data
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_collection_schedules_updated_at BEFORE UPDATE ON collection_schedules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_notification_config_updated_at BEFORE UPDATE ON notification_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default notification templates
INSERT INTO notification_templates (alert_type, subject_template, body_template) VALUES
('LOW_RATE',
 '[CASCADE] ⚠ Collection Rate Alert - {rate} hrs/day',
 'The data collection rate has dropped below target:

Current Rate: {rate} hours/day
Target Rate: 110 hours/day
Active Stations: {stations}

This may delay completion of the 200,000 hour goal.
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

-- Insert default collection schedules for 6 HF bands
INSERT INTO collection_schedules (name, frequency_hz, band_name, priority, target_hours) VALUES
('80m Collection', 3576000, '80m', 1, 33333),
('40m Collection', 7080000, '40m', 1, 33333),
('20m Collection', 14080000, '20m', 1, 33334),
('15m Collection', 21080000, '15m', 2, 33333),
('10m Collection', 28080000, '10m', 2, 33333),
('6m Collection', 50303000, '6m', 3, 33334);