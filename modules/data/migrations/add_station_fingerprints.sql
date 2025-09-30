-- T073: Station fingerprints database schema
-- Tracks anonymous station characteristics for propagation analysis
-- All callsigns are hashed, grid squares preserved for distance calculations

-- Create station_fingerprints table
CREATE TABLE IF NOT EXISTS station_fingerprints (
    id SERIAL PRIMARY KEY,
    station_hash VARCHAR(64) UNIQUE NOT NULL,  -- SHA256 hash of callsign

    -- Temporal tracking
    first_seen TIMESTAMP WITH TIME ZONE NOT NULL,
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL,
    total_observations INTEGER DEFAULT 0,

    -- Frequency characteristics
    primary_bands TEXT[],  -- Array of band names ['20m', '40m']
    frequency_stability_ppm FLOAT,
    frequency_drift_hz_per_min FLOAT,

    -- Signal characteristics
    avg_snr_db FLOAT,
    snr_variance FLOAT,
    typical_power_dbm FLOAT,

    -- Activity patterns (stored as arrays for flexibility)
    active_hours_utc INTEGER[],  -- Hours when active [0-23]
    active_days INTEGER[],  -- Days of week [0-6, 0=Monday]
    duty_cycle FLOAT,  -- Percentage active time

    -- Technical signature
    phase_noise_db FLOAT,
    imd3_db FLOAT,  -- 3rd order intermodulation
    keying_profile JSONB,  -- Rise/fall times, shape
    modulation_quality FLOAT,

    -- Geographic data (grid squares preserved)
    grid_squares TEXT[],  -- All observed grids
    primary_grid VARCHAR(6),  -- Most common grid

    -- Behavioral patterns
    message_types JSONB,  -- {"CQ": 100, "QSO": 250}
    qso_duration_avg_min FLOAT,
    response_time_avg_sec FLOAT,

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Indexes for common queries
    INDEX idx_station_hash ON station_fingerprints(station_hash),
    INDEX idx_primary_grid ON station_fingerprints(primary_grid),
    INDEX idx_last_seen ON station_fingerprints(last_seen),
    INDEX idx_primary_bands ON station_fingerprints USING GIN(primary_bands)
);

-- Create persistent_paths table for TX-RX pair tracking
CREATE TABLE IF NOT EXISTS persistent_paths (
    id SERIAL PRIMARY KEY,
    tx_hash VARCHAR(64) NOT NULL,  -- Transmitter station hash
    rx_hash VARCHAR(64) NOT NULL,  -- Receiver station hash

    -- Path characteristics
    tx_grid VARCHAR(6),  -- TX grid square (preserved)
    rx_grid VARCHAR(6),  -- RX grid square (preserved)
    distance_km FLOAT,
    bearing_degrees FLOAT,

    -- Signal statistics
    observation_count INTEGER DEFAULT 0,
    avg_snr_db FLOAT,
    max_snr_db FLOAT,
    min_snr_db FLOAT,
    snr_variance FLOAT,

    -- Propagation modes observed
    modes_observed JSONB,  -- {"F2": 50, "Es": 10, "TEP": 5}

    -- Temporal patterns
    best_hours_utc INTEGER[],  -- Best propagation hours
    seasonal_pattern JSONB,  -- Monthly statistics

    -- Success metrics
    decode_success_rate FLOAT,  -- Percentage of successful decodes
    bidirectional BOOLEAN DEFAULT FALSE,  -- Path works both ways

    -- Timestamps
    first_observed TIMESTAMP WITH TIME ZONE NOT NULL,
    last_observed TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Composite primary key for uniqueness
    UNIQUE(tx_hash, rx_hash),

    -- Indexes
    INDEX idx_tx_hash ON persistent_paths(tx_hash),
    INDEX idx_rx_hash ON persistent_paths(rx_hash),
    INDEX idx_distance ON persistent_paths(distance_km),
    INDEX idx_last_observed ON persistent_paths(last_observed)
);

-- Create station_observations table for raw data
CREATE TABLE IF NOT EXISTS station_observations (
    id BIGSERIAL PRIMARY KEY,
    station_hash VARCHAR(64) NOT NULL,

    -- Signal data
    frequency BIGINT NOT NULL,  -- Frequency in Hz
    band VARCHAR(10),
    mode VARCHAR(10),  -- FT8, WSPR, etc.
    snr_db FLOAT,
    drift_hz FLOAT,

    -- Message content (anonymized)
    message_type VARCHAR(20),  -- CQ, QSO, BEACON
    grid_square VARCHAR(6),  -- Preserved for analysis

    -- IQ sample reference
    recording_id UUID,  -- Reference to recording session
    sample_offset INTEGER,  -- Offset in recording

    -- Timestamp
    observed_at TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Indexes
    INDEX idx_obs_station ON station_observations(station_hash),
    INDEX idx_obs_time ON station_observations(observed_at),
    INDEX idx_obs_band ON station_observations(band)
);

-- Create aggregated statistics view
CREATE OR REPLACE VIEW station_statistics AS
SELECT
    COUNT(DISTINCT station_hash) as total_stations,
    COUNT(DISTINCT primary_grid) as unique_grids,
    AVG(total_observations) as avg_observations_per_station,
    AVG(duty_cycle) as avg_duty_cycle,
    array_agg(DISTINCT unnest(primary_bands)) as all_bands_used,
    MAX(last_seen) as latest_activity,
    MIN(first_seen) as earliest_observation
FROM station_fingerprints;

-- Create path statistics view
CREATE OR REPLACE VIEW path_statistics AS
SELECT
    COUNT(*) as total_paths,
    AVG(distance_km) as avg_distance_km,
    MAX(distance_km) as max_distance_km,
    AVG(observation_count) as avg_observations_per_path,
    SUM(CASE WHEN bidirectional THEN 1 ELSE 0 END) as bidirectional_paths,
    AVG(decode_success_rate) as avg_success_rate
FROM persistent_paths;

-- Function to update station fingerprint
CREATE OR REPLACE FUNCTION update_station_fingerprint(
    p_station_hash VARCHAR,
    p_observation JSONB
) RETURNS VOID AS $$
DECLARE
    v_existing_id INTEGER;
BEGIN
    -- Check if fingerprint exists
    SELECT id INTO v_existing_id
    FROM station_fingerprints
    WHERE station_hash = p_station_hash;

    IF v_existing_id IS NULL THEN
        -- Create new fingerprint
        INSERT INTO station_fingerprints (
            station_hash,
            first_seen,
            last_seen,
            total_observations,
            avg_snr_db,
            primary_grid
        ) VALUES (
            p_station_hash,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP,
            1,
            (p_observation->>'snr')::FLOAT,
            p_observation->>'grid'
        );
    ELSE
        -- Update existing fingerprint
        UPDATE station_fingerprints
        SET
            last_seen = CURRENT_TIMESTAMP,
            total_observations = total_observations + 1,
            avg_snr_db = (avg_snr_db * total_observations + (p_observation->>'snr')::FLOAT) / (total_observations + 1),
            updated_at = CURRENT_TIMESTAMP
        WHERE id = v_existing_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to record path observation
CREATE OR REPLACE FUNCTION record_path_observation(
    p_tx_hash VARCHAR,
    p_rx_hash VARCHAR,
    p_tx_grid VARCHAR,
    p_rx_grid VARCHAR,
    p_snr FLOAT,
    p_mode VARCHAR
) RETURNS VOID AS $$
DECLARE
    v_distance FLOAT;
    v_existing_id INTEGER;
BEGIN
    -- Calculate distance (simplified - would use proper great circle)
    -- This is a placeholder - real implementation would use PostGIS
    v_distance := 100; -- Placeholder

    -- Check if path exists
    SELECT id INTO v_existing_id
    FROM persistent_paths
    WHERE tx_hash = p_tx_hash AND rx_hash = p_rx_hash;

    IF v_existing_id IS NULL THEN
        -- Create new path
        INSERT INTO persistent_paths (
            tx_hash, rx_hash,
            tx_grid, rx_grid,
            distance_km,
            observation_count,
            avg_snr_db,
            max_snr_db,
            min_snr_db,
            first_observed,
            last_observed
        ) VALUES (
            p_tx_hash, p_rx_hash,
            p_tx_grid, p_rx_grid,
            v_distance,
            1,
            p_snr,
            p_snr,
            p_snr,
            CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP
        );
    ELSE
        -- Update existing path
        UPDATE persistent_paths
        SET
            observation_count = observation_count + 1,
            avg_snr_db = (avg_snr_db * observation_count + p_snr) / (observation_count + 1),
            max_snr_db = GREATEST(max_snr_db, p_snr),
            min_snr_db = LEAST(min_snr_db, p_snr),
            last_observed = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = v_existing_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update fingerprint on observation insert
CREATE OR REPLACE FUNCTION trigger_update_fingerprint()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM update_station_fingerprint(
        NEW.station_hash,
        jsonb_build_object(
            'snr', NEW.snr_db,
            'grid', NEW.grid_square,
            'frequency', NEW.frequency
        )
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_fingerprint_on_observation
AFTER INSERT ON station_observations
FOR EACH ROW
EXECUTE FUNCTION trigger_update_fingerprint();

-- Privacy safeguards
COMMENT ON TABLE station_fingerprints IS 'Anonymous station profiles - callsigns are hashed, grid squares preserved for propagation analysis';
COMMENT ON COLUMN station_fingerprints.station_hash IS 'SHA256 hash of callsign with salt - irreversible';
COMMENT ON COLUMN station_fingerprints.primary_grid IS 'Grid square preserved in cleartext for distance calculations';

-- Grant appropriate permissions
GRANT SELECT ON station_statistics TO CASCADE_readonly;
GRANT SELECT ON path_statistics TO CASCADE_readonly;
GRANT SELECT, INSERT, UPDATE ON station_fingerprints TO CASCADE_app;
GRANT SELECT, INSERT, UPDATE ON persistent_paths TO CASCADE_app;
GRANT SELECT, INSERT ON station_observations TO CASCADE_app;