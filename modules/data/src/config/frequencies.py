"""Frequency configuration for KiwiSDR data collection.

Implements FR-021: Specific center frequencies for optimal QRN and propagation capture.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class BandConfig:
    """Configuration for a single HF band."""

    name: str
    center_khz: int
    bandwidth_khz: int = 12  # Standard 12 kHz IQ bandwidth
    wspr_freq_khz: float | None = None
    ft8_freq_khz: float | None = None
    quiet_zone: Tuple[int, int] | None = None  # (start_khz, end_khz)
    description: str = ""


# FR-021: Specific center frequencies to maximize quiet spectrum coverage
# while capturing propagation indicators (FT8/WSPR)
BAND_CONFIGS: Dict[str, BandConfig] = {
    "80m": BandConfig(
        name="80m",
        center_khz=3576,
        wspr_freq_khz=3568.6,
        ft8_freq_khz=3573.0,
        quiet_zone=(3576, 3582),
        description="Captures WSPR 3568.6, FT8 3573, quiet zone 3576-3582"
    ),
    "40m": BandConfig(
        name="40m",
        center_khz=7080,
        ft8_freq_khz=7074.0,
        quiet_zone=(7078, 7086),
        description="Captures FT8 7074, quiet digital sub-band 7078-7086"
    ),
    "20m": BandConfig(
        name="20m",
        center_khz=14080,
        ft8_freq_khz=14074.0,
        quiet_zone=(14078, 14086),
        description="Captures FT8 14074, quiet zone 14078-14086"
    ),
    "15m": BandConfig(
        name="15m",
        center_khz=21080,
        ft8_freq_khz=21074.0,
        quiet_zone=(21078, 21086),
        description="Captures FT8 21074, quiet zone 21078-21086"
    ),
    "10m": BandConfig(
        name="10m",
        center_khz=28080,
        ft8_freq_khz=28074.0,
        quiet_zone=(28078, 28086),
        description="Captures FT8 28074, quiet zone 28078-28086"
    ),
    "6m": BandConfig(
        name="6m",
        center_khz=50303,
        wspr_freq_khz=50293.0,
        quiet_zone=(50297, 50309),
        description="Captures WSPR 50293, quiet zone 50297-50309"
    ),
}

# All configured bands for easy iteration
BANDS = list(BAND_CONFIGS.keys())
DEFAULT_BANDS = ["20m", "40m", "80m"]  # Most active bands for baseline collection


def get_frequency_range(band: str) -> Tuple[int, int]:
    """Get the frequency range for a band based on center and bandwidth.

    Args:
        band: Band name (e.g., "20m")

    Returns:
        Tuple of (start_khz, end_khz)
    """
    config = BAND_CONFIGS.get(band)
    if not config:
        raise ValueError(f"Unknown band: {band}")

    half_bw = config.bandwidth_khz // 2
    return (config.center_khz - half_bw, config.center_khz + half_bw)


def get_quiet_zones(band: str) -> List[Tuple[int, int]]:
    """Get quiet zones within a band's coverage.

    Args:
        band: Band name

    Returns:
        List of (start_khz, end_khz) tuples for quiet zones
    """
    config = BAND_CONFIGS.get(band)
    if not config or not config.quiet_zone:
        return []
    return [config.quiet_zone]


def get_signal_frequencies(band: str) -> Dict[str, float]:
    """Get FT8/WSPR frequencies for a band.

    Args:
        band: Band name

    Returns:
        Dict with 'ft8' and/or 'wspr' keys and frequency values
    """
    config = BAND_CONFIGS.get(band)
    if not config:
        return {}

    freqs = {}
    if config.ft8_freq_khz:
        freqs['ft8'] = config.ft8_freq_khz
    if config.wspr_freq_khz:
        freqs['wspr'] = config.wspr_freq_khz

    return freqs