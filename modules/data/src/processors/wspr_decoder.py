"""WSPR signal decoder for weak signal propagation analysis.

Implements T030: WSPR decoder (FR-026).
"""

import asyncio
import logging
import numpy as np
import subprocess
import tempfile
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

from scipy import signal
from scipy.io import wavfile

from ..config import config
from ..config.frequencies import BAND_CONFIGS
from ..models import SessionLocal, WSPRSignal, PropagationRecord

logger = logging.getLogger(__name__)


@dataclass
class WSPRMessage:
    """Decoded WSPR message."""

    timestamp: datetime
    frequency_hz: float
    snr_db: float
    drift_hz: float
    power_dbm: int
    callsign_hash: str
    grid_square: str
    distance_km: float
    mode: str = "WSPR"


@dataclass
class WSPRSpot:
    """WSPR propagation spot."""

    timestamp: datetime
    band: str
    tx_callsign_hash: str
    tx_grid: str
    rx_callsign_hash: str
    rx_grid: str
    frequency_mhz: float
    power_dbm: int
    snr_db: float
    drift_hz: float
    distance_km: float
    bearing_degrees: float
    propagation_mode: str


class WSPRDecoder:
    """Decoder for WSPR weak signals with propagation tracking."""

    def __init__(self):
        """Initialize WSPR decoder."""
        self.db = SessionLocal()
        self.wspr_executable = self._find_wspr_decoder()

        # WSPR parameters
        self.symbol_time = 8192.0 / 12000  # ~0.683 seconds
        self.message_duration = 162 * self.symbol_time  # ~110.6 seconds
        self.tone_spacing = 12000.0 / 8192  # ~1.46 Hz

    def _find_wspr_decoder(self) -> str:
        """Find WSPR decoder executable.

        Returns:
            Path to decoder
        """
        candidates = [
            "/usr/local/bin/wsprd",
            "/usr/bin/wsprd",
            "wsprd",
            "/opt/wspr/wsprd",
        ]

        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "-h"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if "WSPR" in result.stderr or result.returncode == 0:
                    logger.info(f"Found WSPR decoder: {candidate}")
                    return candidate
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        logger.warning("No WSPR decoder found, using built-in detection")
        return ""

    async def process_iq_data(
        self,
        iq_data: np.ndarray,
        sample_rate: int,
        center_frequency_khz: float,
        band: str,
        start_time: datetime,
    ) -> List[WSPRMessage]:
        """Process IQ data for WSPR signals.

        Args:
            iq_data: Complex IQ data
            sample_rate: Sample rate in Hz
            center_frequency_khz: Center frequency
            band: Band name
            start_time: Recording start time

        Returns:
            List of decoded WSPR messages
        """
        logger.info(f"Processing {len(iq_data)} IQ samples for WSPR on {band}")

        # Get WSPR frequency for this band
        wspr_freq_hz = self._get_wspr_frequency(band)
        if not wspr_freq_hz:
            logger.warning(f"No WSPR frequency defined for {band}")
            return []

        center_freq_hz = center_frequency_khz * 1000
        freq_offset_hz = wspr_freq_hz - center_freq_hz

        # Extract WSPR audio window
        audio_data = await self._extract_wspr_audio(
            iq_data, sample_rate, freq_offset_hz
        )

        if audio_data is None:
            return []

        # Process 2-minute WSPR intervals (starting at even minutes)
        interval_samples = int(120 * 12000)  # 2 minutes at 12kHz
        messages = []

        # Align to WSPR timing (even minutes UTC)
        start_offset = self._calculate_wspr_alignment(start_time)
        start_sample = int(start_offset * 12000)

        for i in range(start_sample, len(audio_data), interval_samples):
            window = audio_data[i : i + interval_samples]
            if len(window) < int(110 * 12000):  # Need at least 110 seconds
                break

            window_start = start_time + timedelta(seconds=i / 12000)

            # Process this 2-minute window
            window_messages = await self._decode_wspr_window(
                window, 12000, wspr_freq_hz, window_start
            )
            messages.extend(window_messages)

        # Anonymize callsigns
        for msg in messages:
            msg.callsign_hash = self._anonymize_callsign(msg.callsign_hash)

        logger.info(f"Decoded {len(messages)} WSPR messages on {band}")
        return messages

    def _get_wspr_frequency(self, band: str) -> Optional[float]:
        """Get WSPR frequency for band.

        Args:
            band: Band name

        Returns:
            WSPR frequency in Hz
        """
        wspr_frequencies = {
            "160m": 1838000,
            "80m": 3570000,
            "40m": 7040000,
            "30m": 10140000,
            "20m": 14097000,
            "17m": 18106000,
            "15m": 21096000,
            "12m": 24926000,
            "10m": 28126000,
            "6m": 50294000,
        }
        return wspr_frequencies.get(band)

    def _calculate_wspr_alignment(self, start_time: datetime) -> float:
        """Calculate offset to align with WSPR timing.

        Args:
            start_time: Recording start time

        Returns:
            Offset in seconds to next even minute
        """
        # WSPR transmissions start at even minutes UTC
        seconds_past_minute = start_time.second + start_time.microsecond / 1e6

        if seconds_past_minute == 0:
            return 0.0  # Already aligned

        # Wait until next even minute
        if start_time.minute % 2 == 0:
            # Current minute is even, wait for next even minute
            return 120.0 - seconds_past_minute
        else:
            # Current minute is odd, wait for next minute (which will be even)
            return 60.0 - seconds_past_minute

    async def _extract_wspr_audio(
        self,
        iq_data: np.ndarray,
        sample_rate: int,
        freq_offset_hz: float,
    ) -> Optional[np.ndarray]:
        """Extract WSPR audio from IQ data.

        Args:
            iq_data: Complex IQ samples
            sample_rate: IQ sample rate
            freq_offset_hz: Frequency offset

        Returns:
            Real audio samples at 12 kHz
        """
        try:
            # Frequency shift
            if freq_offset_hz != 0:
                t = np.arange(len(iq_data)) / sample_rate
                shift_signal = np.exp(-1j * 2 * np.pi * freq_offset_hz * t)
                iq_data = iq_data * shift_signal

            # Low-pass filter for WSPR bandwidth (~200 Hz)
            nyquist = sample_rate / 2
            cutoff = min(100, nyquist * 0.8)  # 200 Hz bandwidth
            sos = signal.butter(8, cutoff / nyquist, btype="low", output="sos")
            filtered = signal.sosfilt(sos, iq_data)

            # Decimate to 12 kHz
            decimation = sample_rate // 12000
            if decimation > 1:
                audio = signal.decimate(filtered, decimation, ftype="fir")
            else:
                audio = filtered

            # Convert to real
            return np.real(audio).astype(np.float32)

        except Exception as e:
            logger.error(f"Error extracting WSPR audio: {e}")
            return None

    async def _decode_wspr_window(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        center_freq_hz: float,
        window_start: datetime,
    ) -> List[WSPRMessage]:
        """Decode WSPR in 2-minute window.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            center_freq_hz: Center frequency
            window_start: Window start time

        Returns:
            WSPR messages
        """
        if self.wspr_executable:
            return await self._decode_with_wsprd(
                audio_data, sample_rate, window_start
            )
        else:
            return await self._detect_wspr_builtin(
                audio_data, sample_rate, center_freq_hz, window_start
            )

    async def _decode_with_wsprd(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        window_start: datetime,
    ) -> List[WSPRMessage]:
        """Decode using wsprd.

        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            window_start: Window start time

        Returns:
            Decoded messages
        """
        messages = []

        try:
            # Create temporary C2 file (wsprd format)
            with tempfile.NamedTemporaryFile(suffix=".c2", delete=False) as temp_file:
                # Convert to wsprd format (complex float32)
                if len(audio_data) >= int(110.25 * sample_rate):
                    # Take exactly 110.25 seconds
                    wspr_samples = int(110.25 * sample_rate)
                    audio_trimmed = audio_data[:wspr_samples]

                    # Convert to complex (real + 0j)
                    complex_data = audio_trimmed.astype(np.complex64)

                    # Write in wsprd format
                    complex_data.tofile(temp_file.name)

                    # Run wsprd
                    cmd = [
                        self.wspr_executable,
                        "-d",  # Deep search
                        "-C", "500",  # CPU seconds
                        temp_file.name,
                    ]

                    result = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=tempfile.gettempdir(),
                    )

                    stdout, stderr = await result.communicate()

                    if result.returncode == 0:
                        # Parse output
                        messages = self._parse_wsprd_output(
                            stdout.decode(), window_start
                        )
                    else:
                        logger.debug(f"wsprd error: {stderr.decode()}")

                # Clean up
                os.unlink(temp_file.name)

        except Exception as e:
            logger.error(f"wsprd execution error: {e}")

        return messages

    async def _detect_wspr_builtin(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        center_freq_hz: float,
        window_start: datetime,
    ) -> List[WSPRMessage]:
        """Built-in WSPR detection.

        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            center_freq_hz: Center frequency
            window_start: Window start time

        Returns:
            Detected signals
        """
        messages = []

        try:
            # Look for WSPR-like signals in 1400-1600 Hz range
            nperseg = int(sample_rate * 1.0)  # 1-second windows
            f, t, Sxx = signal.spectrogram(
                audio_data, sample_rate, nperseg=nperseg, noverlap=nperseg // 2
            )

            # WSPR frequency range (1400-1600 Hz)
            wspr_mask = (f >= 1400) & (f <= 1600)
            wspr_spectrum = Sxx[wspr_mask, :]

            # Look for consistent signals over ~110 seconds
            min_duration_bins = int(100 / (t[1] - t[0]))  # 100+ seconds

            # Find persistent signals
            threshold = np.percentile(wspr_spectrum, 90)
            strong_mask = wspr_spectrum > threshold

            # Check for signals lasting minimum duration
            for freq_idx in range(len(f[wspr_mask])):
                signal_times = strong_mask[freq_idx, :]

                # Find continuous segments
                segments = self._find_continuous_segments(signal_times)

                for start_idx, end_idx in segments:
                    duration_bins = end_idx - start_idx

                    if duration_bins >= min_duration_bins:
                        frequency = f[wspr_mask][freq_idx]
                        start_time = window_start + timedelta(seconds=t[start_idx])

                        # Estimate signal parameters
                        signal_power = np.mean(wspr_spectrum[freq_idx, start_idx:end_idx])
                        noise_power = np.median(wspr_spectrum)
                        snr_db = 10 * np.log10(signal_power / max(noise_power, 1e-12))

                        if snr_db > -30:  # WSPR can decode very weak signals
                            messages.append(
                                WSPRMessage(
                                    timestamp=start_time,
                                    frequency_hz=center_freq_hz + frequency,
                                    snr_db=snr_db,
                                    drift_hz=0.0,  # Can't measure without decoding
                                    power_dbm=20,  # Default assumption
                                    callsign_hash="[DETECTED]",
                                    grid_square="XX00",
                                    distance_km=0.0,
                                    mode="WSPR_DETECT",
                                )
                            )

        except Exception as e:
            logger.error(f"WSPR detection error: {e}")

        return messages

    def _find_continuous_segments(self, signal_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous segments in boolean mask.

        Args:
            signal_mask: Boolean array

        Returns:
            List of (start_idx, end_idx) tuples
        """
        segments = []
        in_segment = False
        start_idx = 0

        for i, value in enumerate(signal_mask):
            if value and not in_segment:
                # Start of segment
                start_idx = i
                in_segment = True
            elif not value and in_segment:
                # End of segment
                segments.append((start_idx, i))
                in_segment = False

        # Handle segment that goes to end
        if in_segment:
            segments.append((start_idx, len(signal_mask)))

        return segments

    def _parse_wsprd_output(
        self, output: str, window_start: datetime
    ) -> List[WSPRMessage]:
        """Parse wsprd output.

        Args:
            output: wsprd stdout
            window_start: Window start time

        Returns:
            Parsed WSPR messages
        """
        messages = []

        for line in output.strip().split("\n"):
            if not line.strip() or line.startswith("#"):
                continue

            try:
                # wsprd output format:
                # time freq snr dt call grid power drift
                parts = line.split()
                if len(parts) < 8:
                    continue

                time_str = parts[0]
                freq_hz = float(parts[1])
                snr_db = float(parts[2])
                dt_seconds = float(parts[3])
                callsign = parts[4]
                grid = parts[5]
                power_dbm = int(parts[6])
                drift_hz = float(parts[7])

                # Parse time (HHMM format)
                hour = int(time_str[:2])
                minute = int(time_str[2:4])

                timestamp = window_start.replace(hour=hour, minute=minute, second=0)

                # Calculate distance (placeholder)
                distance_km = self._calculate_distance(grid)

                messages.append(
                    WSPRMessage(
                        timestamp=timestamp,
                        frequency_hz=freq_hz,
                        snr_db=snr_db,
                        drift_hz=drift_hz,
                        power_dbm=power_dbm,
                        callsign_hash=callsign,  # Will be anonymized later
                        grid_square=grid,
                        distance_km=distance_km,
                    )
                )

            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse wsprd line: {line} ({e})")

        return messages

    def _calculate_distance(self, grid_square: str) -> float:
        """Calculate distance from grid square.

        Args:
            grid_square: 4 or 6 character grid square

        Returns:
            Distance in km (placeholder)
        """
        # Placeholder - would use proper grid square calculation
        return abs(hash(grid_square)) % 20000

    def _anonymize_callsign(self, callsign: str) -> str:
        """Anonymize WSPR callsign.

        Args:
            callsign: Original callsign

        Returns:
            Anonymized hash
        """
        if not callsign or callsign in ["[DETECTED]", ""]:
            return callsign

        salt = config.CALLSIGN_SALT or "cascade_wspr_salt"
        hash_input = f"{callsign}:{salt}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:12]

    async def store_wspr_signals(
        self, messages: List[WSPRMessage], session_id: str, band: str
    ):
        """Store WSPR signals in database.

        Args:
            messages: WSPR messages
            session_id: Recording session ID
            band: Band name
        """
        try:
            for msg in messages:
                signal_record = WSPRSignal(
                    session_id=session_id,
                    timestamp=msg.timestamp,
                    frequency_hz=msg.frequency_hz,
                    snr_db=msg.snr_db,
                    drift_hz=msg.drift_hz,
                    power_dbm=msg.power_dbm,
                    callsign_hash=msg.callsign_hash,
                    grid_square=msg.grid_square,
                    distance_km=msg.distance_km,
                    band=band,
                    mode=msg.mode,
                )

                self.db.add(signal_record)

            self.db.commit()
            logger.info(f"Stored {len(messages)} WSPR signals for session {session_id}")

        except Exception as e:
            logger.error(f"Error storing WSPR signals: {e}")
            self.db.rollback()

    async def create_propagation_spots(
        self, messages: List[WSPRMessage], band: str, rx_grid: str = "XX00"
    ) -> List[WSPRSpot]:
        """Create propagation spots from WSPR messages.

        Args:
            messages: WSPR messages
            band: Band name
            rx_grid: Receiver grid square

        Returns:
            WSPR propagation spots
        """
        spots = []

        for msg in messages:
            if msg.grid_square and msg.grid_square != "XX00":
                # Calculate propagation parameters
                distance_km = msg.distance_km
                bearing = self._calculate_bearing(msg.grid_square, rx_grid)
                prop_mode = self._classify_wspr_propagation(
                    distance_km, msg.snr_db, band, msg.timestamp
                )

                spot = WSPRSpot(
                    timestamp=msg.timestamp,
                    band=band,
                    tx_callsign_hash=msg.callsign_hash,
                    tx_grid=msg.grid_square,
                    rx_callsign_hash="RX_STATION",  # Our receiver
                    rx_grid=rx_grid,
                    frequency_mhz=msg.frequency_hz / 1e6,
                    power_dbm=msg.power_dbm,
                    snr_db=msg.snr_db,
                    drift_hz=msg.drift_hz,
                    distance_km=distance_km,
                    bearing_degrees=bearing,
                    propagation_mode=prop_mode,
                )

                spots.append(spot)

        return spots

    def _calculate_bearing(self, tx_grid: str, rx_grid: str) -> float:
        """Calculate bearing between grid squares.

        Args:
            tx_grid: Transmitter grid
            rx_grid: Receiver grid

        Returns:
            Bearing in degrees (placeholder)
        """
        # Placeholder calculation
        return (hash(tx_grid) + hash(rx_grid)) % 360

    def _classify_wspr_propagation(
        self, distance_km: float, snr_db: float, band: str, timestamp: datetime
    ) -> str:
        """Classify WSPR propagation mode.

        Args:
            distance_km: Distance
            snr_db: SNR
            band: Band
            timestamp: Time

        Returns:
            Propagation mode
        """
        # Enhanced classification for WSPR
        hour = timestamp.hour

        if distance_km < 500:
            return "NVIS"
        elif distance_km < 1500:
            if 6 <= hour <= 18:  # Daytime
                return "E_LAYER"
            else:
                return "F2_SHORT"
        elif distance_km < 4000:
            return "F2_MEDIUM"
        elif distance_km < 8000:
            return "F2_LONG"
        elif distance_km < 15000:
            return "F2_DX"
        else:
            if snr_db > -25:
                return "EME"  # Possible Earth-Moon-Earth
            else:
                return "F2_EXTREME"

    def close(self):
        """Close decoder resources."""
        if self.db:
            self.db.close()


async def process_wspr_file(file_path: str, band: str) -> Dict[str, Any]:
    """Process WSPR from audio file.

    Args:
        file_path: Audio file path
        band: Band name

    Returns:
        Processing results
    """
    decoder = WSPRDecoder()

    try:
        sample_rate, audio_data = wavfile.read(file_path)

        # Convert to complex if needed
        if audio_data.dtype != np.complex64:
            iq_data = audio_data.astype(np.float32)
            if len(iq_data.shape) == 2:
                iq_data = iq_data[:, 0] + 1j * iq_data[:, 1]
            else:
                iq_data = iq_data + 1j * np.zeros_like(iq_data)
        else:
            iq_data = audio_data

        # Get center frequency for band
        wspr_freq = decoder._get_wspr_frequency(band)
        center_freq_khz = wspr_freq / 1000 if wspr_freq else 14097

        # Process
        messages = await decoder.process_iq_data(
            iq_data, sample_rate, center_freq_khz, band, datetime.utcnow()
        )

        return {
            "file": file_path,
            "band": band,
            "sample_rate": sample_rate,
            "duration_seconds": len(iq_data) / sample_rate,
            "messages_decoded": len(messages),
            "messages": [asdict(msg) for msg in messages[:5]],
        }

    finally:
        decoder.close()