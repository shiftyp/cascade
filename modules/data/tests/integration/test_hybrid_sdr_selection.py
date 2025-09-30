"""
Integration tests for hybrid SDR selection algorithm
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import uuid

# These imports will fail initially (TDD approach)
from cascade_collector.collectors.hybrid_sdr_selector import HybridSDRSelector
from cascade_collector.models.sdr_source import SDRSource
from cascade_collector.collectors.kiwi_client import KiwiClient
from cascade_collector.collectors.websdr_client import WebSDRClient


class TestHybridSDRSelection:
    """Test hybrid KiwiSDR/WebSDR selection algorithm"""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session"""
        session = Mock()
        session.query = Mock()
        session.commit = Mock()
        session.rollback = Mock()
        return session

    @pytest.fixture
    def mixed_sdr_pool(self):
        """Create mixed pool of KiwiSDR and WebSDR sources"""
        sdrs = []

        # KiwiSDRs with limited daily usage
        for i in range(3):
            kiwi = Mock(spec=SDRSource)
            kiwi.sdr_id = str(uuid.uuid4())
            kiwi.sdr_type = 'KIWISDR'
            kiwi.url = f"kiwisdr{i}.example.com:8073"
            kiwi.name = f"KiwiSDR {i}"
            kiwi.daily_limit_minutes = 90
            kiwi.usage_today_minutes = i * 30
            kiwi.session_limit_minutes = 30
            kiwi.is_active = True
            kiwi.reliability_score = 0.85 - (i * 0.05)
            kiwi.grid_square = f"FN{i}0"
            sdrs.append(kiwi)

        # WebSDRs with higher capacity
        for i in range(2):
            websdr = Mock(spec=SDRSource)
            websdr.sdr_id = str(uuid.uuid4())
            websdr.sdr_type = 'WEBSDR'
            websdr.url = f"websdr{i}.university.edu"
            websdr.name = f"University WebSDR {i}"
            websdr.institution_type = 'UNIVERSITY'
            websdr.daily_limit_minutes = 0  # Unlimited for research
            websdr.session_limit_minutes = 3600  # 1 hour sessions
            websdr.is_active = True
            websdr.reliability_score = 0.95 - (i * 0.02)
            websdr.grid_square = f"JO{i}0"
            sdrs.append(websdr)

        return sdrs

    @pytest.mark.asyncio
    async def test_prefers_websdr_for_long_sessions(self, mock_db_session, mixed_sdr_pool):
        """Test that WebSDR is preferred for long recording sessions"""
        selector = HybridSDRSelector(mock_db_session)
        mock_db_session.query().filter().all.return_value = mixed_sdr_pool

        # Request 2-hour session
        selected = await selector.select_optimal_sdr(
            band="20m",
            duration_minutes=120,
            prefer_type='auto'
        )

        # Should select WebSDR for long session
        assert selected is not None
        assert selected.sdr_type == 'WEBSDR'
        assert selected.session_limit_minutes >= 120

    @pytest.mark.asyncio
    async def test_uses_kiwisdr_for_geographic_coverage(self, mock_db_session, mixed_sdr_pool):
        """Test that KiwiSDRs are used when geographic diversity is needed"""
        selector = HybridSDRSelector(mock_db_session)
        mock_db_session.query().filter().all.return_value = mixed_sdr_pool

        # Request specific geographic area
        selected = await selector.select_optimal_sdr(
            band="40m",
            duration_minutes=20,
            grid_square_prefix='FN',
            prefer_type='auto'
        )

        # Should select KiwiSDR in FN grid
        assert selected is not None
        assert selected.sdr_type == 'KIWISDR'
        assert selected.grid_square.startswith('FN')

    @pytest.mark.asyncio
    async def test_balances_load_across_types(self, mock_db_session, mixed_sdr_pool):
        """Test that load is balanced between SDR types"""
        selector = HybridSDRSelector(mock_db_session)
        mock_db_session.query().filter().all.return_value = mixed_sdr_pool

        # Make multiple selections
        selections = []
        for _ in range(10):
            sdr = await selector.select_optimal_sdr(
                band="20m",
                duration_minutes=15
            )
            if sdr:
                selections.append(sdr.sdr_type)

        # Should have mix of both types
        kiwi_count = selections.count('KIWISDR')
        websdr_count = selections.count('WEBSDR')
        assert kiwi_count > 0 and websdr_count > 0

    @pytest.mark.asyncio
    async def test_respects_institutional_policies(self, mock_db_session, mixed_sdr_pool):
        """Test that WebSDR institutional policies are respected"""
        selector = HybridSDRSelector(mock_db_session)

        # Add research-restricted WebSDR
        restricted = Mock(spec=SDRSource)
        restricted.sdr_type = 'WEBSDR'
        restricted.institution_type = 'RESEARCH_INSTITUTE'
        restricted.usage_policy = 'RESEARCH_AGREEMENT'
        restricted.research_approved = False
        restricted.is_active = True

        pool_with_restricted = mixed_sdr_pool + [restricted]
        mock_db_session.query().filter().all.return_value = pool_with_restricted

        # Should not select research-restricted SDR without approval
        selected = await selector.select_optimal_sdr(
            band="20m",
            exclude_restricted=True
        )

        assert selected is not None
        assert not (selected.usage_policy == 'RESEARCH_AGREEMENT' and not selected.research_approved)

    @pytest.mark.asyncio
    async def test_fallback_strategy_on_type_failure(self, mock_db_session, mixed_sdr_pool):
        """Test fallback from WebSDR to KiwiSDR on failure"""
        selector = HybridSDRSelector(mock_db_session)
        mock_db_session.query().filter().all.return_value = mixed_sdr_pool

        # Simulate WebSDR failures
        with patch.object(selector, 'connect_websdr', side_effect=ConnectionError):
            selected = await selector.select_with_fallback(
                band="20m",
                prefer_type='WEBSDR'
            )

            # Should fallback to KiwiSDR
            assert selected is not None
            assert selected.sdr_type == 'KIWISDR'

    @pytest.mark.asyncio
    async def test_optimal_selection_for_event_scaling(self, mock_db_session, mixed_sdr_pool):
        """Test optimal SDR selection during space weather events"""
        selector = HybridSDRSelector(mock_db_session)
        mock_db_session.query().filter().all.return_value = mixed_sdr_pool

        # Simulate storm mode requiring many SDRs
        storm_selections = await selector.select_for_event(
            event_type='GEOMAGNETIC_STORM',
            k_index=7,
            required_stations=8
        )

        # Should use both types to meet demand
        sdr_types = [s.sdr_type for s in storm_selections]
        assert 'KIWISDR' in sdr_types
        assert 'WEBSDR' in sdr_types
        assert len(storm_selections) <= len(mixed_sdr_pool)

    @pytest.mark.asyncio
    async def test_bandwidth_based_selection(self, mock_db_session, mixed_sdr_pool):
        """Test selection based on bandwidth requirements"""
        selector = HybridSDRSelector(mock_db_session)
        mock_db_session.query().filter().all.return_value = mixed_sdr_pool

        # Request wide bandwidth (>12 kHz)
        selected = await selector.select_optimal_sdr(
            band="20m",
            bandwidth_khz=48,
            prefer_type='auto'
        )

        # Should prefer WebSDR for wide bandwidth
        assert selected is not None
        assert selected.sdr_type == 'WEBSDR'  # WebSDRs support wider bandwidth

    @pytest.mark.asyncio
    async def test_cost_optimization_strategy(self, mock_db_session, mixed_sdr_pool):
        """Test that selection optimizes for cost/usage limits"""
        selector = HybridSDRSelector(mock_db_session)

        # Set KiwiSDRs near daily limit
        for sdr in mixed_sdr_pool:
            if sdr.sdr_type == 'KIWISDR':
                sdr.usage_today_minutes = 85  # Near 90 min limit

        mock_db_session.query().filter().all.return_value = mixed_sdr_pool

        # Should prefer WebSDR to preserve KiwiSDR quota
        selected = await selector.select_optimal_sdr(
            band="40m",
            duration_minutes=30,
            optimize_quota=True
        )

        assert selected is not None
        assert selected.sdr_type == 'WEBSDR'

    @pytest.mark.asyncio
    async def test_multi_channel_capability_selection(self, mock_db_session, mixed_sdr_pool):
        """Test selection for multi-channel recording capability"""
        selector = HybridSDRSelector(mock_db_session)
        mock_db_session.query().filter().all.return_value = mixed_sdr_pool

        # Request multi-channel recording
        selected = await selector.select_for_multichannel(
            channels=[
                {'freq': 3573000, 'bw': 12000},
                {'freq': 7074000, 'bw': 12000},
            ]
        )

        # Should select SDR capable of multiple channels
        assert selected is not None
        # WebSDRs typically support multi-channel better
        assert selected.sdr_type == 'WEBSDR'

    @pytest.mark.asyncio
    async def test_historical_performance_weighting(self, mock_db_session, mixed_sdr_pool):
        """Test that historical performance influences selection"""
        selector = HybridSDRSelector(mock_db_session)

        # Add performance history
        for sdr in mixed_sdr_pool:
            sdr.success_rate = 0.95 if sdr.sdr_type == 'WEBSDR' else 0.85
            sdr.avg_connection_time = 2.0 if sdr.sdr_type == 'WEBSDR' else 5.0

        mock_db_session.query().filter().all.return_value = mixed_sdr_pool

        # Should consider performance in selection
        selected = await selector.select_optimal_sdr(
            band="20m",
            weight_performance=True
        )

        assert selected is not None
        assert selected.success_rate >= 0.9