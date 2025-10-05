-- CASCADE KiwiSDR Data Collector - Complete Initial Schema
-- Generated from SQLAlchemy models
-- Date: 2025-10-01
--
-- This is the complete initial schema for fresh database setup.
-- Run this on an empty database to create all tables.

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- ENUM Types
-- =============================================================================

CREATE TYPE solarcyclephase AS ENUM ('MINIMUM', 'ASCENDING', 'MAXIMUM', 'DESCENDING');
CREATE TYPE qbophase AS ENUM ('WESTERLY', 'EASTERLY', 'TRANSITION');
CREATE TYPE season AS ENUM ('WINTER', 'SPRING', 'SUMMER', 'AUTUMN');
CREATE TYPE institutiontype AS ENUM ('INDIVIDUAL', 'UNIVERSITY', 'RESEARCH_INSTITUTE', 'AMATEUR_CLUB');

-- =============================================================================
-- SDR Source Tables
-- =============================================================================

-- KiwiSDR Sources
CREATE TABLE kiwisdr_sources (
	kiwisdr_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	url VARCHAR(255) NOT NULL,
	name VARCHAR(255),
	grid_square VARCHAR(10),
	latitude FLOAT,
	longitude FLOAT,
	timezone VARCHAR(50) DEFAULT 'UTC',
	min_freq_khz FLOAT NOT NULL DEFAULT 10,
	max_freq_khz FLOAT NOT NULL DEFAULT 30000,
	max_users INTEGER NOT NULL DEFAULT 4,
	has_gps BOOLEAN NOT NULL DEFAULT TRUE,
	antenna_type VARCHAR(255),
	peak_hours_utc JSON,
	owner_contact TEXT,
	has_research_agreement BOOLEAN NOT NULL DEFAULT FALSE,
	usage_policy_notes TEXT,
	daily_limit_minutes FLOAT NOT NULL DEFAULT 90,
	daily_usage_minutes FLOAT NOT NULL DEFAULT 0,
	last_usage_reset TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	total_usage_minutes FLOAT NOT NULL DEFAULT 0,
	last_connected TIMESTAMP WITH TIME ZONE,
	last_seen TIMESTAMP WITH TIME ZONE,
	active BOOLEAN NOT NULL DEFAULT TRUE,
	reliability_score FLOAT,
	failure_count INTEGER NOT NULL DEFAULT 0,
	consecutive_failures INTEGER NOT NULL DEFAULT 0,
	last_failure_type VARCHAR(50),
	last_failure_time TIMESTAMP WITH TIME ZONE,
	potentially_blacklisted BOOLEAN NOT NULL DEFAULT FALSE,
	requires_auth BOOLEAN NOT NULL DEFAULT FALSE,
	auth_config JSON,
	notes TEXT,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	PRIMARY KEY (kiwisdr_id),
	UNIQUE (url)
);

COMMENT ON TABLE kiwisdr_sources IS 'Public KiwiSDR receivers for data collection';
COMMENT ON COLUMN kiwisdr_sources.grid_square IS 'Maidenhead grid square (up to 10 chars for extended locators)';
COMMENT ON COLUMN kiwisdr_sources.latitude IS 'Latitude (rounded for privacy)';
COMMENT ON COLUMN kiwisdr_sources.longitude IS 'Longitude (rounded for privacy)';
COMMENT ON COLUMN kiwisdr_sources.daily_limit_minutes IS 'Daily usage limit in minutes (typically 90 for KiwiSDR)';
COMMENT ON COLUMN kiwisdr_sources.daily_usage_minutes IS 'Current daily usage in minutes';
COMMENT ON COLUMN kiwisdr_sources.consecutive_failures IS 'Consecutive failures without success (for blacklist detection)';
COMMENT ON COLUMN kiwisdr_sources.last_failure_type IS 'Type of last failure: timeout, refused, auth_failed, blacklist';
COMMENT ON COLUMN kiwisdr_sources.potentially_blacklisted IS 'Flag indicating possible IP blacklist (10+ consecutive connection refused)';

-- WebSDR Sources
CREATE TABLE websdr_sources (
	websdr_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	url VARCHAR(255) NOT NULL,
	name VARCHAR(255) NOT NULL,
	institution_type institutiontype NOT NULL,
	institution_name VARCHAR(255),
	owner_contact TEXT,
	contact_email VARCHAR(255),
	has_research_agreement BOOLEAN NOT NULL DEFAULT FALSE,
	agreement_details JSON,
	agreement_expiry TIMESTAMP WITH TIME ZONE,
	extended_usage_allowed BOOLEAN NOT NULL DEFAULT FALSE,
	grid_square VARCHAR(6),
	latitude FLOAT,
	longitude FLOAT,
	timezone VARCHAR(50) DEFAULT 'UTC',
	min_freq_khz FLOAT NOT NULL DEFAULT 10,
	max_freq_khz FLOAT NOT NULL DEFAULT 30000,
	max_users INTEGER NOT NULL DEFAULT 100,
	bandwidth_khz FLOAT NOT NULL DEFAULT 12,
	daily_limit_minutes INTEGER,
	session_limit_minutes INTEGER DEFAULT 180,
	peak_hours_utc JSON,
	usage_policy_notes TEXT,
	daily_usage_minutes FLOAT NOT NULL DEFAULT 0,
	last_usage_reset TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	total_usage_minutes FLOAT NOT NULL DEFAULT 0,
	last_connected TIMESTAMP WITH TIME ZONE,
	active BOOLEAN NOT NULL DEFAULT TRUE,
	reliability_score FLOAT,
	failure_count INTEGER NOT NULL DEFAULT 0,
	preferred_for_long_sessions BOOLEAN NOT NULL DEFAULT TRUE,
	requires_auth BOOLEAN NOT NULL DEFAULT FALSE,
	auth_config JSON,
	antenna_description TEXT,
	receiver_description TEXT,
	notes TEXT,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	PRIMARY KEY (websdr_id),
	UNIQUE (url)
);

COMMENT ON TABLE websdr_sources IS 'WebSDR receivers (typically institutional) for extended collection sessions';

-- =============================================================================
-- Recording Sessions
-- =============================================================================

CREATE TABLE recording_sessions (
	session_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	kiwisdr_id UUID,
	websdr_id UUID,
	correlation_id UUID NOT NULL,
	start_time TIMESTAMP WITH TIME ZONE NOT NULL,
	end_time TIMESTAMP WITH TIME ZONE,
	duration_seconds INTEGER,
	frequency_khz FLOAT NOT NULL,
	bandwidth_khz FLOAT NOT NULL DEFAULT 12,
	sample_rate INTEGER NOT NULL DEFAULT 12000,
	mode VARCHAR(10) NOT NULL DEFAULT 'iq',
	propagation_mode VARCHAR(20),
	gps_locked BOOLEAN NOT NULL DEFAULT FALSE,
	avg_snr_db FLOAT,
	sample_rate_accuracy FLOAT,
	gaps_detected INTEGER NOT NULL DEFAULT 0,
	file_path TEXT,
	tigris_path TEXT,
	file_size_bytes INTEGER,
	compressed BOOLEAN NOT NULL DEFAULT FALSE,
	status VARCHAR(20) NOT NULL DEFAULT 'recording',
	error_message TEXT,
	processing_status VARCHAR(20) NOT NULL DEFAULT 'unprocessed',
	processing_version INTEGER,
	processing_completed_at TIMESTAMP WITH TIME ZONE,
	ft8_extracted BOOLEAN NOT NULL DEFAULT FALSE,
	wspr_extracted BOOLEAN NOT NULL DEFAULT FALSE,
	qrn_analyzed BOOLEAN NOT NULL DEFAULT FALSE,
	processing_metadata TEXT,
	band VARCHAR(10),
	notes TEXT,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	PRIMARY KEY (session_id),
	FOREIGN KEY(kiwisdr_id) REFERENCES kiwisdr_sources (kiwisdr_id) ON DELETE CASCADE,
	FOREIGN KEY(websdr_id) REFERENCES websdr_sources (websdr_id) ON DELETE CASCADE,
	CHECK (status IN ('pending', 'recording', 'completed', 'failed', 'cancelled')),
	CHECK (processing_status IN ('unprocessed', 'processing', 'processed', 'failed'))
);

COMMENT ON TABLE recording_sessions IS 'IQ recording sessions from SDRs';
COMMENT ON COLUMN recording_sessions.correlation_id IS 'Links simultaneous recordings from multiple SDRs';
COMMENT ON COLUMN recording_sessions.processing_status IS 'unprocessed = raw only, processed = FT8/WSPR/QRN extracted';

-- =============================================================================
-- Extracted Signals
-- =============================================================================

-- QRN (Noise) Samples
CREATE TABLE qrn_samples (
	sample_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	session_id UUID NOT NULL,
	correlation_id UUID NOT NULL,
	timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
	frequency_khz FLOAT NOT NULL,
	bandwidth_khz FLOAT NOT NULL DEFAULT 2.5,
	duration_seconds FLOAT NOT NULL,
	avg_power_dbm FLOAT,
	peak_power_dbm FLOAT,
	noise_floor_dbm FLOAT,
	snr_db FLOAT,
	quiet_periods JSON,
	quiet_percentage FLOAT,
	has_quiet_zones BOOLEAN NOT NULL DEFAULT FALSE,
	grid_square VARCHAR(6),
	geographic_region VARCHAR(50),
	qrn_type VARCHAR(50),
	impulsiveness FLOAT,
	spectral_occupancy FLOAT,
	channel_data JSON,
	file_path TEXT,
	file_size_bytes INTEGER,
	quality_score FLOAT,
	validated BOOLEAN NOT NULL DEFAULT FALSE,
	notes TEXT,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	PRIMARY KEY (sample_id),
	FOREIGN KEY(session_id) REFERENCES recording_sessions (session_id) ON DELETE CASCADE
);

COMMENT ON TABLE qrn_samples IS 'Atmospheric noise samples for training';
COMMENT ON COLUMN qrn_samples.channel_data IS 'Multi-channel QRN data (9x 2.5kHz channels)';

-- Propagation Records (FT8/WSPR combined)
CREATE TABLE propagation_records (
	record_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	session_id UUID NOT NULL,
	correlation_id UUID NOT NULL,
	timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
	frequency_hz INTEGER NOT NULL,
	mode VARCHAR(10) NOT NULL,
	snr_db FLOAT NOT NULL,
	drift_hz FLOAT,
	signal_strength_dbm FLOAT,
	tx_grid VARCHAR(6),
	rx_grid VARCHAR(6),
	distance_km FLOAT,
	azimuth_degrees FLOAT,
	propagation_mode VARCHAR(50),
	propagation_confidence FLOAT,
	mode_indicators JSON,
	solar_flux FLOAT,
	k_index INTEGER,
	callsign_hash VARCHAR(64),
	message_type VARCHAR(20),
	decode_confidence FLOAT,
	false_decode_probability FLOAT,
	notes TEXT,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	PRIMARY KEY (record_id),
	FOREIGN KEY(session_id) REFERENCES recording_sessions (session_id) ON DELETE CASCADE,
	CHECK (mode IN ('FT8', 'WSPR', 'FT4'))
);

COMMENT ON TABLE propagation_records IS 'Decoded propagation signals (FT8/WSPR) with anonymized callsigns';
COMMENT ON COLUMN propagation_records.callsign_hash IS 'One-way hash of callsign for privacy';

-- FT8 Signals (detailed)
CREATE TABLE ft8_signals (
	signal_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	session_id UUID NOT NULL,
	timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
	frequency_hz FLOAT NOT NULL,
	snr_db FLOAT NOT NULL,
	dt_seconds FLOAT,
	message_hash VARCHAR(32),
	grid_square VARCHAR(6),
	band VARCHAR(10),
	mode VARCHAR(20) NOT NULL DEFAULT 'FT8',
	raw_message VARCHAR(50),
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	PRIMARY KEY (signal_id),
	FOREIGN KEY(session_id) REFERENCES recording_sessions (session_id) ON DELETE CASCADE
);

-- WSPR Signals (detailed)
CREATE TABLE wspr_signals (
	signal_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	session_id UUID NOT NULL,
	timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
	frequency_hz FLOAT NOT NULL,
	snr_db FLOAT NOT NULL,
	drift_hz FLOAT,
	power_dbm INTEGER,
	callsign_hash VARCHAR(32),
	grid_square VARCHAR(6),
	distance_km FLOAT,
	band VARCHAR(10),
	mode VARCHAR(20) NOT NULL DEFAULT 'WSPR',
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	PRIMARY KEY (signal_id),
	FOREIGN KEY(session_id) REFERENCES recording_sessions (session_id) ON DELETE CASCADE
);

-- Atmospheric Events
CREATE TABLE atmospheric_events (
	event_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	session_id UUID NOT NULL,
	timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
	event_type VARCHAR(20) NOT NULL,
	peak_amplitude FLOAT,
	duration_ms FLOAT,
	rise_time_us FLOAT,
	decay_time_ms FLOAT,
	avg_noise_level FLOAT,
	min_noise_level FLOAT,
	quality_score FLOAT,
	frequency_content JSON,
	classification VARCHAR(50),
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	PRIMARY KEY (event_id),
	FOREIGN KEY(session_id) REFERENCES recording_sessions (session_id) ON DELETE CASCADE,
	CHECK (event_type IN ('IMPULSE', 'BURST', 'CONTINUOUS', 'OTHER'))
);

-- =============================================================================
-- Space Weather Data
-- =============================================================================

CREATE TABLE space_weather_data (
	weather_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	observation_time TIMESTAMP WITH TIME ZONE NOT NULL,
	solar_flux FLOAT,
	sunspot_number INTEGER,
	xray_class VARCHAR(5),
	xray_flux FLOAT,
	xray_flare_start TIMESTAMP WITH TIME ZONE,
	xray_flare_peak TIMESTAMP WITH TIME ZONE,
	k_index INTEGER,
	a_index INTEGER,
	dst_index INTEGER,
	solar_wind_speed FLOAT,
	solar_wind_density FLOAT,
	bz_component FLOAT,
	muf_3000 FLOAT,
	fof2 FLOAT,
	storm_level VARCHAR(10),
	aurora_visible FLOAT,
	raw_data JSON,
	solar_cycle_phase solarcyclephase,
	solar_cycle_number INTEGER,
	qbo_index FLOAT,
	qbo_phase qbophase,
	lunar_phase FLOAT,
	lunar_age_days INTEGER,
	season season,
	seasonal_balance_factor FLOAT,
	equinoctial_enhancement BOOLEAN NOT NULL DEFAULT FALSE,
	cycle_metadata JSON,
	collection_window_factor FLOAT,
	opportunity_limited_mode BOOLEAN NOT NULL DEFAULT FALSE,
	rarity_multiplier FLOAT,
	source VARCHAR(50) NOT NULL,
	created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
	PRIMARY KEY (weather_id),
	CHECK (k_index >= 0 AND k_index <= 9)
);

COMMENT ON TABLE space_weather_data IS 'Space weather conditions for propagation analysis';

-- =============================================================================
-- Collection Management
-- =============================================================================

-- Collection Schedules
CREATE TABLE collection_schedules (
	schedule_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	name VARCHAR(255) NOT NULL,
	active BOOLEAN DEFAULT TRUE,
	start_time TIME WITHOUT TIME ZONE,
	end_time TIME WITHOUT TIME ZONE,
	days_of_week JSON,
	frequency_khz INTEGER NOT NULL,
	band VARCHAR(10),
	duration_seconds INTEGER DEFAULT 360,
	interval_minutes INTEGER,
	preferred_sdrs JSON,
	min_sdrs INTEGER DEFAULT 1,
	max_sdrs INTEGER,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	PRIMARY KEY (schedule_id)
);

-- Collection Alerts
CREATE TABLE collection_alerts (
	alert_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	alert_type VARCHAR(50) NOT NULL,
	severity VARCHAR(20) NOT NULL,
	message TEXT NOT NULL,
	details TEXT,
	acknowledged BOOLEAN DEFAULT FALSE,
	acknowledged_by VARCHAR(255),
	acknowledged_at TIMESTAMP WITH TIME ZONE,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	PRIMARY KEY (alert_id),
	CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'))
);

-- Notification Configs
CREATE TABLE notification_configs (
	config_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	name VARCHAR(255) NOT NULL,
	active BOOLEAN DEFAULT TRUE,
	email_enabled BOOLEAN DEFAULT FALSE,
	email_addresses JSON,
	min_sdr_threshold INTEGER,
	max_failure_count INTEGER,
	storage_threshold_gb INTEGER,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	PRIMARY KEY (config_id)
);

-- Notification Templates
CREATE TABLE notification_templates (
	template_id UUID NOT NULL DEFAULT uuid_generate_v4(),
	name VARCHAR(255) NOT NULL,
	alert_type VARCHAR(50) NOT NULL,
	subject VARCHAR(255) NOT NULL,
	body TEXT NOT NULL,
	created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
	PRIMARY KEY (template_id),
	UNIQUE (name)
);

-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- KiwiSDR indexes
CREATE INDEX idx_kiwisdr_active ON kiwisdr_sources(active) WHERE active = TRUE;
CREATE INDEX idx_kiwisdr_potentially_blacklisted ON kiwisdr_sources(potentially_blacklisted) WHERE potentially_blacklisted = TRUE;
CREATE INDEX idx_kiwisdr_last_failure ON kiwisdr_sources(last_failure_type, last_failure_time);
CREATE INDEX idx_kiwisdr_grid_square ON kiwisdr_sources(grid_square);

-- Recording session indexes
CREATE INDEX ix_recording_sessions_correlation_id ON recording_sessions(correlation_id);
CREATE INDEX idx_recording_sessions_status ON recording_sessions(status);
CREATE INDEX idx_recording_sessions_processing_status ON recording_sessions(processing_status);
CREATE INDEX idx_recording_sessions_start_time ON recording_sessions(start_time);
CREATE INDEX idx_recording_sessions_band ON recording_sessions(band);
CREATE INDEX idx_recording_sessions_kiwisdr ON recording_sessions(kiwisdr_id);
CREATE INDEX idx_recording_sessions_websdr ON recording_sessions(websdr_id);

-- QRN sample indexes
CREATE INDEX ix_qrn_samples_correlation_id ON qrn_samples(correlation_id);
CREATE INDEX ix_qrn_samples_timestamp ON qrn_samples(timestamp);
CREATE INDEX idx_qrn_samples_session ON qrn_samples(session_id);

-- Propagation record indexes
CREATE INDEX ix_propagation_records_correlation_id ON propagation_records(correlation_id);
CREATE INDEX ix_propagation_records_timestamp ON propagation_records(timestamp);
CREATE INDEX ix_propagation_records_mode ON propagation_records(mode);
CREATE INDEX idx_propagation_records_session ON propagation_records(session_id);

-- FT8/WSPR indexes
CREATE INDEX ix_ft8_signals_timestamp ON ft8_signals(timestamp);
CREATE INDEX ix_ft8_signals_session_id ON ft8_signals(session_id);
CREATE INDEX ix_wspr_signals_timestamp ON wspr_signals(timestamp);
CREATE INDEX ix_wspr_signals_session_id ON wspr_signals(session_id);

-- Atmospheric event indexes
CREATE INDEX ix_atmospheric_events_session_id ON atmospheric_events(session_id);
CREATE INDEX ix_atmospheric_events_timestamp ON atmospheric_events(timestamp);

-- Space weather indexes
CREATE UNIQUE INDEX ix_space_weather_data_observation_time ON space_weather_data(observation_time);
CREATE INDEX idx_space_weather_k_index ON space_weather_data(k_index);
CREATE INDEX idx_space_weather_solar_flux ON space_weather_data(solar_flux);

-- =============================================================================
-- Complete
-- =============================================================================
