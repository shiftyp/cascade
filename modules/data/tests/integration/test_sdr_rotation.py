"""Integration test for KiwiSDR connection rotation with usage limit enforcement.

Tests FR-008 and FR-014: SDR rotation and usage limit respect.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.collectors.sdr_manager import SDRManager
from src.models import SessionLocal
from src.models.kiwisdr_source import KiwiSDRSource


@pytest.fixture
async def db_session():
    """Create test database session."""
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
async def sdr_manager(db_session):
    """Create SDR manager with test database."""
    return SDRManager(db_session)


@pytest.fixture
async def test_sdrs(db_session):
    """Create test SDRs in database."""
    sdrs = []

    # Create 3 test SDRs with different usage levels
    sdr1 = KiwiSDRSource(
        url="test1.kiwisdr.com:8073",
        name="Test SDR 1",
        grid_square="FN31",
        daily_usage_minutes=85,  # Near limit
        active=True,
        reliability_score=0.9
    )

    sdr2 = KiwiSDRSource(
        url="test2.kiwisdr.com:8073",
        name="Test SDR 2",
        grid_square="JO22",
        daily_usage_minutes=0,  # Fresh
        active=True,
        reliability_score=0.9
    )

    sdr3 = KiwiSDRSource(
        url="test3.kiwisdr.com:8073",
        name="Test SDR 3",
        grid_square="PM95",
        daily_usage_minutes=90,  # At limit
        active=True,
        reliability_score=0.9
    )

    db_session.add_all([sdr1, sdr2, sdr3])
    db_session.commit()

    sdrs = [sdr1, sdr2, sdr3]
    yield sdrs

    # Cleanup
    for sdr in sdrs:
        db_session.delete(sdr)
    db_session.commit()


class TestSDRRotation:
    """Integration tests for SDR rotation with usage limits."""

    @pytest.mark.asyncio
    async def test_rotate_sdrs_on_limit(self, sdr_manager, test_sdrs):
        """Test SDR rotation when daily limit is approached (FR-008, FR-014)."""
        sdr1, sdr2, sdr3 = test_sdrs

        # Act - Get next available
        available = await sdr_manager.get_next_available()

        # Assert - Should get sdr2 (0 min used) or sdr1 (85 min, still has 5 min)
        # Should NOT get sdr3 (90 min, at limit)
        assert available is not None
        assert available.kiwisdr_id != sdr3.kiwisdr_id
        assert available.remaining_daily_minutes >= 5

        # Act - Update sdr1 to exceed limit
        if available.kiwisdr_id == sdr1.kiwisdr_id:
            await sdr_manager.update_usage(sdr1, minutes=10)
            # Now sdr1 is at 95 minutes (over limit)

            # Get next available again
            next_available = await sdr_manager.get_next_available()

            # Assert - Should now only get sdr2
            assert next_available.kiwisdr_id == sdr2.kiwisdr_id

    @pytest.mark.asyncio
    async def test_usage_tracking_enforcement(self, sdr_manager, test_sdrs):
        """Test that usage is properly tracked and enforced."""
        sdr1, sdr2, sdr3 = test_sdrs

        # Get fresh SDR
        sdr = await sdr_manager.get_next_available()
        assert sdr.kiwisdr_id == sdr2.kiwisdr_id  # Should be the one with 0 usage

        initial_usage = sdr.daily_usage_minutes

        # Simulate 10 minutes of usage
        await sdr_manager.update_usage(sdr, minutes=10)

        # Verify usage was tracked
        assert sdr.daily_usage_minutes == initial_usage + 10
        assert sdr.remaining_daily_minutes == 90 - (initial_usage + 10)

    @pytest.mark.asyncio
    async def test_concurrent_sdr_selection(self, sdr_manager, test_sdrs):
        """Test selecting multiple SDRs for concurrent use."""
        # Act - Request 2 concurrent SDRs
        concurrent = await sdr_manager.get_concurrent_sdrs(count=2)

        # Assert - Should get 2 SDRs (sdr1 and sdr2, not sdr3 which is at limit)
        assert len(concurrent) == 2
        assert all(sdr.remaining_daily_minutes >= 5 for sdr in concurrent)

        # Verify all have different URLs (no duplicates)
        urls = [sdr.url for sdr in concurrent]
        assert len(urls) == len(set(urls))

    @pytest.mark.asyncio
    async def test_daily_usage_reset(self, sdr_manager, test_sdrs):
        """Test daily usage counter reset."""
        sdr1, sdr2, sdr3 = test_sdrs

        # Set last reset to yesterday
        sdr3.last_usage_reset = datetime.utcnow() - timedelta(days=1, hours=2)
        sdr_manager.db.commit()

        # Trigger reset check
        await sdr_manager.check_and_reset_usage()

        # Verify sdr3 is now available
        sdr3_refreshed = sdr_manager.db.query(KiwiSDRSource).filter_by(kiwisdr_id=sdr3.kiwisdr_id).first()
        assert sdr3_refreshed.daily_usage_minutes == 0
        assert sdr3_refreshed.remaining_daily_minutes == 90

    @pytest.mark.asyncio
    async def test_sdr_becomes_unavailable_at_limit(self, sdr_manager, test_sdrs):
        """Test that SDR is removed from pool when limit reached."""
        sdr1, sdr2, sdr3 = test_sdrs

        # sdr3 is already at limit
        # Try to get it
        available = await sdr_manager.get_next_available()

        # Should not get sdr3
        assert available.kiwisdr_id != sdr3.kiwisdr_id

    @pytest.mark.asyncio
    async def test_least_used_sdr_selected_first(self, sdr_manager, test_sdrs):
        """Test that SDR with least usage is selected first (FR-008)."""
        sdr1, sdr2, sdr3 = test_sdrs

        # sdr2 has 0 minutes, sdr1 has 85 minutes
        available = await sdr_manager.get_next_available()

        # Should select sdr2 (least used)
        assert available.kiwisdr_id == sdr2.kiwisdr_id
        assert available.daily_usage_minutes == 0