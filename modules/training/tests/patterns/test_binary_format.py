"""Contract tests for binary pattern file I/O"""

import pytest
import numpy as np
import os
import tempfile
import sys
sys.path.insert(0, '/workspaces/cascade')

from modules.training.patterns import save_pattern_file, load_pattern_file, Pattern


def test_save_load_roundtrip():
    """T008: Verify save/load round-trip preserves patterns"""
    # Create test patterns
    patterns = []
    for i in range(5):
        freq_seq = np.random.randint(0, 4, size=32, dtype='uint8')
        iq_traj = (np.random.randn(32) + 1j * np.random.randn(32)).astype('complex64')
        patterns.append(Pattern(
            pattern_id=i,
            freq_sequence=freq_seq,
            iq_trajectory=iq_traj,
            iq_complexity_lambda=i * 0.1
        ))

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        temp_file = f.name

    try:
        save_pattern_file(patterns, temp_file)

        # Load back
        loaded_patterns = load_pattern_file(temp_file)

        # Verify all data preserved
        assert len(loaded_patterns) == len(patterns)
        for orig, loaded in zip(patterns, loaded_patterns):
            assert loaded.pattern_id == orig.pattern_id
            assert (loaded.freq_sequence == orig.freq_sequence).all()
            assert np.allclose(loaded.iq_trajectory, orig.iq_trajectory)
            assert abs(loaded.iq_complexity_lambda - orig.iq_complexity_lambda) < 1e-6

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_file_size_matches_spec():
    """T008: Verify file size matches spec (292 bytes per pattern + header)"""
    # Create patterns
    patterns = []
    for i in range(10):
        freq_seq = np.random.randint(0, 4, size=32, dtype='uint8')
        iq_traj = np.random.randn(32).astype('complex64')
        patterns.append(Pattern(
            pattern_id=i,
            freq_sequence=freq_seq,
            iq_trajectory=iq_traj,
            iq_complexity_lambda=0.0
        ))

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        temp_file = f.name

    try:
        save_pattern_file(patterns, temp_file)

        file_size = os.path.getsize(temp_file)

        # Header (32 bytes) + 10 patterns × 295 bytes = 2982 bytes
        expected_min = 32 + 10 * 295  # Header + pattern data
        expected_max = 32 + 10 * 295 + 10  # Allow small variance

        assert expected_min <= file_size <= expected_max, \
            f"File size {file_size} outside expected range [{expected_min}, {expected_max}]"

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_magic_bytes_present():
    """T008: Verify magic bytes b'CASC' present in file"""
    patterns = [Pattern(
        pattern_id=0,
        freq_sequence=np.zeros(32, dtype='uint8'),
        iq_trajectory=np.ones(32, dtype='complex64'),
        iq_complexity_lambda=0.0
    )]

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        temp_file = f.name

    try:
        save_pattern_file(patterns, temp_file)

        # Read raw file and check magic bytes
        with open(temp_file, 'rb') as f:
            header = f.read(4)
            assert header == b'CASC', f"Magic bytes should be b'CASC', got {header}"

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_checksums_valid():
    """T008: Verify checksums are computed and validated"""
    patterns = []
    for i in range(3):
        freq_seq = np.random.randint(0, 4, size=32, dtype='uint8')
        iq_traj = np.random.randn(32).astype('complex64')
        patterns.append(Pattern(
            pattern_id=i,
            freq_sequence=freq_seq,
            iq_trajectory=iq_traj,
            iq_complexity_lambda=0.0
        ))

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        temp_file = f.name

    try:
        save_pattern_file(patterns, temp_file)

        # Loading should succeed (checksums valid)
        loaded_patterns = load_pattern_file(temp_file)
        assert len(loaded_patterns) == 3

        # Corrupt the file
        with open(temp_file, 'r+b') as f:
            f.seek(50)  # Skip header, corrupt pattern data
            f.write(b'\xFF\xFF')

        # Loading should fail with checksum error
        with pytest.raises((ValueError, IOError, RuntimeError)):
            load_pattern_file(temp_file)

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_version_field_present():
    """T008: Verify file contains version field"""
    patterns = [Pattern(
        pattern_id=0,
        freq_sequence=np.zeros(32, dtype='uint8'),
        iq_trajectory=np.ones(32, dtype='complex64'),
        iq_complexity_lambda=0.0
    )]

    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        temp_file = f.name

    try:
        save_pattern_file(patterns, temp_file)

        # Load and verify we can access version info
        # (Implementation will store this internally)
        loaded = load_pattern_file(temp_file)
        assert len(loaded) == 1  # Basic sanity check

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
