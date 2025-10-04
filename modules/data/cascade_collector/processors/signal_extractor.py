"""Signal extractor for CASCADE - combines FT8, WSPR, and QRN processing."""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

from .ft8_processor import FT8Processor, FT8Message
from .wspr_processor import WSPRProcessor, WSPRMessage
from .qrn_processor import QRNProcessor, QRNCharacteristics

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSignals:
    """Container for all extracted signals from a recording."""
    timestamp: datetime
    frequency: float
    bandwidth: float
    ft8_messages: List[FT8Message]
    wspr_messages: List[WSPRMessage]
    qrn_characteristics: QRNCharacteristics
    propagation_data: Dict[str, Any]
    metadata: Dict[str, Any]


class SignalExtractor:
    """Extracts all signal types from recordings."""

    def __init__(self):
        """Initialize signal extractor with all processors."""
        self.ft8_processor = FT8Processor()
        self.wspr_processor = WSPRProcessor()
        self.qrn_processor = QRNProcessor()
        self.recordings_processed = 0

    async def extract_all(self, audio_data: np.ndarray,
                         sample_rate: int,
                         frequency: float,
                         bandwidth: float = 12000) -> ExtractedSignals:
        """Extract all signal types from recording.

        Args:
            audio_data: Audio samples (IQ or real)
            sample_rate: Sample rate in Hz
            frequency: Center frequency in Hz
            bandwidth: Recording bandwidth in Hz

        Returns:
            ExtractedSignals with all decoded data
        """
        self.recordings_processed += 1

        # Process FT8
        ft8_messages = await self.ft8_processor.process_recording(
            audio_data, sample_rate, frequency
        )

        # Process WSPR (only if recording is long enough)
        wspr_messages = []
        if len(audio_data) >= sample_rate * 110:  # At least 110 seconds for WSPR
            wspr_messages = await self.wspr_processor.process_recording(
                audio_data, sample_rate, frequency
            )

        # Analyze QRN
        qrn_characteristics = await self.qrn_processor.process_recording(
            audio_data, sample_rate, frequency, bandwidth
        )

        # Combine propagation data
        propagation_data = self._combine_propagation_data(
            ft8_messages, wspr_messages, qrn_characteristics
        )

        # Create metadata
        metadata = {
            "sample_rate": sample_rate,
            "samples": len(audio_data),
            "duration_seconds": len(audio_data) / sample_rate,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }

        extracted = ExtractedSignals(
            timestamp=datetime.now(timezone.utc),
            frequency=frequency,
            bandwidth=bandwidth,
            ft8_messages=ft8_messages,
            wspr_messages=wspr_messages,
            qrn_characteristics=qrn_characteristics,
            propagation_data=propagation_data,
            metadata=metadata
        )

        logger.info(f"Extracted {len(ft8_messages)} FT8, {len(wspr_messages)} WSPR, "
                   f"QRN type: {qrn_characteristics.noise_type}")

        return extracted

    def _combine_propagation_data(self, ft8_messages: List[FT8Message],
                                 wspr_messages: List[WSPRMessage],
                                 qrn: QRNCharacteristics) -> Dict[str, Any]:
        """Combine propagation data from all sources.

        Args:
            ft8_messages: FT8 messages
            wspr_messages: WSPR messages
            qrn: QRN characteristics

        Returns:
            Combined propagation analysis
        """
        data = {
            "ft8_propagation": self.ft8_processor.extract_propagation_data(ft8_messages),
            "wspr_propagation": self.wspr_processor.extract_propagation_data(wspr_messages),
            "noise_conditions": {
                "type": qrn.noise_type,
                "floor_dbm": qrn.noise_floor_dbm,
                "peak_dbm": qrn.peak_power_dbm,
                "impulse_rate": qrn.impulse_rate,
                "spectral_occupancy": qrn.spectral_occupancy
            }
        }

        # Calculate combined metrics
        all_grids = set()
        if data["ft8_propagation"]:
            all_grids.update(data["ft8_propagation"].get("grid_squares", []))
        if data["wspr_propagation"]:
            all_grids.update(data["wspr_propagation"].get("grid_squares", []))

        data["total_grids"] = len(all_grids)
        data["grid_squares"] = list(all_grids)

        return data

    def get_statistics(self) -> Dict[str, Any]:
        """Get extractor statistics.

        Returns:
            Combined statistics from all processors
        """
        return {
            "recordings_processed": self.recordings_processed,
            "ft8_stats": self.ft8_processor.get_statistics(),
            "wspr_stats": self.wspr_processor.get_statistics(),
            "qrn_stats": self.qrn_processor.get_statistics()
        }