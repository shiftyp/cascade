-- Migration: Add blacklist detection and error categorization fields
-- Date: 2025-10-01
-- Purpose: Track connection failures by type and detect potential blacklisting

-- Add new columns to kiwisdr_sources
ALTER TABLE kiwisdr_sources
ADD COLUMN IF NOT EXISTS consecutive_failures INTEGER NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_failure_type VARCHAR(50),
ADD COLUMN IF NOT EXISTS last_failure_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS potentially_blacklisted BOOLEAN NOT NULL DEFAULT FALSE;

-- Add comments
COMMENT ON COLUMN kiwisdr_sources.consecutive_failures IS 'Consecutive failures without success (for blacklist detection)';
COMMENT ON COLUMN kiwisdr_sources.last_failure_type IS 'Type of last failure: timeout, refused, auth_failed, blacklist';
COMMENT ON COLUMN kiwisdr_sources.last_failure_time IS 'Time of last connection failure';
COMMENT ON COLUMN kiwisdr_sources.potentially_blacklisted IS 'Flag indicating possible IP blacklist (10+ consecutive connection refused)';

-- Create index for blacklist queries
CREATE INDEX IF NOT EXISTS idx_kiwisdr_potentially_blacklisted
ON kiwisdr_sources(potentially_blacklisted)
WHERE potentially_blacklisted = TRUE;

-- Create index for failure type queries
CREATE INDEX IF NOT EXISTS idx_kiwisdr_last_failure
ON kiwisdr_sources(last_failure_type, last_failure_time);
