-- CASCADE Dashboard SQL Views
-- Provides aggregated views for monitoring data collection

-- Drop existing views if they exist
DROP VIEW IF EXISTS v_collection_status CASCADE;
DROP VIEW IF EXISTS v_hourly_collection_stats CASCADE;
DROP VIEW IF EXISTS v_sdr_performance CASCADE;
DROP VIEW IF EXISTS v_propagation_summary CASCADE;
DROP VIEW IF EXISTS v_qrn_coverage CASCADE;
DROP VIEW IF EXISTS v_space_weather_events CASCADE;
DROP VIEW IF EXISTS v_storage_usage CASCADE;
DROP VIEW IF EXISTS v_qa_sample_quality CASCADE;
DROP VIEW IF EXISTS v_correlation_completeness CASCADE;
DROP VIEW IF EXISTS v_band_coverage CASCADE;

-- Overall collection status view
CREATE VIEW v_collection_status AS
SELECT
    COUNT(DISTINCT rs.id) as total_sessions,
    COUNT(DISTINCT rs.kiwisdr_source_id) as unique_sdrs,
    SUM(EXTRACT(EPOCH FROM (rs.end_time - rs.start_time)) / 3600) as total_hours_collected,
    AVG(EXTRACT(EPOCH FROM (rs.end_time - rs.start_time)) / 3600) as avg_session_hours,
    MIN(rs.start_time) as first_collection,
    MAX(rs.end_time) as last_collection,
    COUNT(DISTINCT DATE(rs.start_time)) as collection_days,
    SUM(rs.file_size_bytes) / (1024^3) as total_storage_gb,
    AVG(rs.sample_rate) as avg_sample_rate,
    COUNT(DISTINCT rs.frequency_band) as bands_covered
FROM recording_sessions rs
WHERE rs.status = 'completed';

-- Hourly collection statistics
CREATE VIEW v_hourly_collection_stats AS
WITH hourly_bins AS (
    SELECT
        date_trunc('hour', rs.start_time) as hour_bin,
        rs.frequency_band,
        COUNT(*) as sessions_started,
        SUM(EXTRACT(EPOCH FROM (
            LEAST(rs.end_time, date_trunc('hour', rs.start_time) + INTERVAL '1 hour') -
            rs.start_time
        )) / 3600) as hours_collected,
        COUNT(DISTINCT rs.kiwisdr_source_id) as active_sdrs
    FROM recording_sessions rs
    WHERE rs.status IN ('completed', 'recording')
    GROUP BY date_trunc('hour', rs.start_time), rs.frequency_band
)
SELECT
    hour_bin,
    frequency_band,
    sessions_started,
    hours_collected,
    active_sdrs,
    SUM(hours_collected) OVER (
        ORDER BY hour_bin
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_hours
FROM hourly_bins
ORDER BY hour_bin DESC;

-- SDR performance metrics
CREATE VIEW v_sdr_performance AS
SELECT
    ks.id as sdr_id,
    ks.name as sdr_name,
    ks.location_grid as grid_square,
    ks.is_active,
    COUNT(rs.id) as total_sessions,
    SUM(EXTRACT(EPOCH FROM (rs.end_time - rs.start_time)) / 3600) as total_hours,
    AVG(EXTRACT(EPOCH FROM (rs.end_time - rs.start_time)) / 60) as avg_session_minutes,
    MAX(rs.end_time) as last_used,
    MIN(rs.start_time) as first_used,
    COUNT(DISTINCT rs.frequency_band) as bands_recorded,
    SUM(CASE WHEN rs.status = 'failed' THEN 1 ELSE 0 END) as failed_sessions,
    ROUND(
        100.0 * SUM(CASE WHEN rs.status = 'completed' THEN 1 ELSE 0 END) /
        NULLIF(COUNT(rs.id), 0),
        2
    ) as success_rate,
    ks.daily_limit_minutes,
    ks.reliability_score
FROM kiwisdr_sources ks
LEFT JOIN recording_sessions rs ON ks.id = rs.kiwisdr_source_id
GROUP BY ks.id, ks.name, ks.location_grid, ks.is_active,
         ks.daily_limit_minutes, ks.reliability_score
ORDER BY total_hours DESC;

-- Propagation data summary
CREATE VIEW v_propagation_summary AS
SELECT
    pr.frequency_band,
    pr.mode,
    pr.propagation_mode,
    COUNT(*) as total_records,
    COUNT(DISTINCT pr.tx_hash) as unique_transmitters,
    COUNT(DISTINCT rs.kiwisdr_source_id) as unique_receivers,
    AVG(pr.snr_db) as avg_snr,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pr.snr_db) as median_snr,
    MIN(pr.snr_db) as min_snr,
    MAX(pr.snr_db) as max_snr,
    AVG(pr.frequency_offset_hz) as avg_freq_offset,
    COUNT(DISTINCT DATE(pr.timestamp)) as days_observed,
    COUNT(DISTINCT pr.correlation_id) as correlated_samples
FROM propagation_records pr
JOIN recording_sessions rs ON pr.recording_session_id = rs.id
GROUP BY pr.frequency_band, pr.mode, pr.propagation_mode
ORDER BY total_records DESC;

-- QRN coverage analysis
CREATE VIEW v_qrn_coverage AS
SELECT
    qs.frequency_band,
    COUNT(*) as total_samples,
    COUNT(DISTINCT rs.kiwisdr_source_id) as unique_locations,
    AVG(qs.noise_floor_dbm) as avg_noise_floor,
    STDDEV(qs.noise_floor_dbm) as noise_floor_stddev,
    AVG(qs.peak_power_dbm) as avg_peak_power,
    SUM(qs.quiet_period_seconds) / 3600.0 as total_quiet_hours,
    AVG(qs.quiet_period_seconds) as avg_quiet_seconds,
    COUNT(DISTINCT qs.correlation_id) as correlated_samples,
    COUNT(DISTINCT DATE(qs.timestamp)) as days_covered,
    MIN(qs.timestamp) as first_sample,
    MAX(qs.timestamp) as last_sample
FROM qrn_samples qs
JOIN recording_sessions rs ON qs.recording_session_id = rs.id
GROUP BY qs.frequency_band
ORDER BY qs.frequency_band;

-- Space weather event correlation
CREATE VIEW v_space_weather_events AS
SELECT
    sw.id,
    sw.timestamp,
    sw.xray_class,
    sw.xray_flux,
    sw.proton_flux,
    sw.electron_flux,
    sw.k_index,
    sw.a_index,
    sw.solar_wind_speed,
    COUNT(DISTINCT rs.id) as concurrent_recordings,
    COUNT(DISTINCT pr.id) as propagation_records,
    COUNT(DISTINCT qs.id) as qrn_samples,
    STRING_AGG(DISTINCT rs.frequency_band, ', ') as bands_recorded
FROM space_weather_data sw
LEFT JOIN recording_sessions rs ON
    rs.start_time <= sw.timestamp + INTERVAL '2 hours' AND
    rs.end_time >= sw.timestamp - INTERVAL '2 hours'
LEFT JOIN propagation_records pr ON
    pr.timestamp BETWEEN sw.timestamp - INTERVAL '2 hours'
    AND sw.timestamp + INTERVAL '2 hours'
LEFT JOIN qrn_samples qs ON
    qs.timestamp BETWEEN sw.timestamp - INTERVAL '2 hours'
    AND sw.timestamp + INTERVAL '2 hours'
WHERE sw.xray_class IS NOT NULL
GROUP BY sw.id, sw.timestamp, sw.xray_class, sw.xray_flux,
         sw.proton_flux, sw.electron_flux, sw.k_index,
         sw.a_index, sw.solar_wind_speed
ORDER BY sw.timestamp DESC;

-- Storage usage breakdown
CREATE VIEW v_storage_usage AS
SELECT
    rs.frequency_band,
    rs.compression_type,
    COUNT(*) as file_count,
    SUM(rs.file_size_bytes) / (1024^3) as total_gb,
    AVG(rs.file_size_bytes) / (1024^2) as avg_file_mb,
    MIN(rs.file_size_bytes) / (1024^2) as min_file_mb,
    MAX(rs.file_size_bytes) / (1024^2) as max_file_mb,
    SUM(rs.file_size_bytes) / NULLIF(SUM(rs.original_size_bytes), 0) as compression_ratio,
    SUM(rs.original_size_bytes - rs.file_size_bytes) / (1024^3) as space_saved_gb
FROM recording_sessions rs
WHERE rs.status = 'completed'
GROUP BY rs.frequency_band, rs.compression_type
ORDER BY total_gb DESC;

-- QA sample quality metrics
CREATE VIEW v_qa_sample_quality AS
SELECT
    qs.frequency_band,
    COUNT(*) as total_samples,
    AVG(qs.quality_score) as avg_quality_score,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY qs.quality_score) as median_quality,
    MIN(qs.quality_score) as min_quality,
    MAX(qs.quality_score) as max_quality,
    SUM(CASE WHEN qs.quality_score < 50 THEN 1 ELSE 0 END) as poor_quality_count,
    SUM(CASE WHEN qs.quality_score >= 80 THEN 1 ELSE 0 END) as high_quality_count,
    COUNT(DISTINCT qs.reviewed_by) as reviewers,
    SUM(CASE WHEN qs.is_quarantined THEN 1 ELSE 0 END) as quarantined_count
FROM qa_samples qs
GROUP BY qs.frequency_band;

-- Correlation completeness analysis
CREATE VIEW v_correlation_completeness AS
WITH correlation_stats AS (
    SELECT
        correlation_id,
        COUNT(DISTINCT 'qrn') as qrn_count,
        COUNT(DISTINCT 'propagation') as prop_count,
        MIN(timestamp) as start_time,
        MAX(timestamp) as end_time,
        COUNT(DISTINCT frequency_band) as band_count
    FROM (
        SELECT correlation_id, timestamp, frequency_band, 'qrn' as type
        FROM qrn_samples
        WHERE correlation_id IS NOT NULL
        UNION ALL
        SELECT correlation_id, timestamp, frequency_band, 'propagation' as type
        FROM propagation_records
        WHERE correlation_id IS NOT NULL
    ) combined
    GROUP BY correlation_id
)
SELECT
    DATE(start_time) as collection_date,
    COUNT(*) as total_correlations,
    SUM(CASE WHEN qrn_count > 0 AND prop_count > 0 THEN 1 ELSE 0 END) as complete_pairs,
    SUM(CASE WHEN qrn_count > 0 AND prop_count = 0 THEN 1 ELSE 0 END) as qrn_only,
    SUM(CASE WHEN qrn_count = 0 AND prop_count > 0 THEN 1 ELSE 0 END) as prop_only,
    AVG(band_count) as avg_bands_per_correlation,
    AVG(EXTRACT(EPOCH FROM (end_time - start_time)) / 60) as avg_duration_minutes
FROM correlation_stats
GROUP BY DATE(start_time)
ORDER BY collection_date DESC;

-- Band coverage heatmap data
CREATE VIEW v_band_coverage AS
WITH hourly_coverage AS (
    SELECT
        date_trunc('hour', rs.start_time) as hour_slot,
        rs.frequency_band,
        SUM(EXTRACT(EPOCH FROM (rs.end_time - rs.start_time)) / 3600) as hours_recorded,
        COUNT(DISTINCT rs.kiwisdr_source_id) as sdr_count
    FROM recording_sessions rs
    WHERE rs.status = 'completed'
    GROUP BY date_trunc('hour', rs.start_time), rs.frequency_band
)
SELECT
    frequency_band,
    EXTRACT(HOUR FROM hour_slot) as hour_of_day,
    AVG(hours_recorded) as avg_hours,
    AVG(sdr_count) as avg_sdrs,
    COUNT(*) as sample_count
FROM hourly_coverage
GROUP BY frequency_band, EXTRACT(HOUR FROM hour_slot)
ORDER BY frequency_band, hour_of_day;

-- Create indexes for better view performance
CREATE INDEX IF NOT EXISTS idx_recording_sessions_status_time
    ON recording_sessions(status, start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_recording_sessions_band
    ON recording_sessions(frequency_band);
CREATE INDEX IF NOT EXISTS idx_propagation_records_correlation
    ON propagation_records(correlation_id);
CREATE INDEX IF NOT EXISTS idx_qrn_samples_correlation
    ON qrn_samples(correlation_id);
CREATE INDEX IF NOT EXISTS idx_space_weather_timestamp
    ON space_weather_data(timestamp);

-- Grant read permissions to dashboard user (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cascade_dashboard') THEN
        GRANT SELECT ON ALL TABLES IN SCHEMA public TO cascade_dashboard;
    END IF;
END $$;