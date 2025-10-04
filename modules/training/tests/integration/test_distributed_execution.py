"""Integration Test: Distributed Execution on Fly.io

Tests Fly.io worker infrastructure for pattern generation.
"""

import pytest
import sys

sys.path.insert(0, '/workspaces/cascade')


@pytest.mark.slow
@pytest.mark.requires_flyio
def test_distributed_execution_small():
    """T040: Test distributed execution with 4 workers

    This is a small-scale test to verify:
    - Workers spawn correctly
    - Results upload to Tigris
    - Coordinator collects and selects best
    - Final pattern file generated

    Expected cost: ~$0.76 (4 workers × 24 hours)
    Expected time: 18-24 hours
    """
    pytest.skip("Requires Fly.io setup and Tigris credentials - run manually")

    # Manual test command:
    # python -m modules.training.patterns generate \
    #     --count 64 \
    #     --distributed \
    #     --workers 4 \
    #     --seed 42


@pytest.mark.slow
@pytest.mark.requires_flyio
def test_distributed_execution_production():
    """Test production distributed execution with 32 workers

    Expected cost: ~$6
    Expected time: 18-24 hours
    Expected quality: -40.2 dB, λ≈0.22
    """
    pytest.skip("Production test - run manually for final pattern generation")


def test_worker_script_exists():
    """Verify worker infrastructure files exist"""
    from pathlib import Path

    worker_dir = Path('/workspaces/cascade/modules/training/fly-pattern-worker')

    assert (worker_dir / 'worker.py').exists()
    assert (worker_dir / 'coordinator.py').exists()
    assert (worker_dir / 'Dockerfile').exists()
    assert (worker_dir / 'fly.toml').exists()
    assert (worker_dir / 'requirements.txt').exists()


def test_worker_requirements_complete():
    """Verify worker has all required dependencies"""
    from pathlib import Path

    req_file = Path('/workspaces/cascade/modules/training/fly-pattern-worker/requirements.txt')
    content = req_file.read_text()

    # Check for critical dependencies
    assert 'numpy' in content
    assert 'scipy' in content
    assert 'boto3' in content  # For Tigris S3
