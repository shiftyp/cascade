"""FLAC compression for CASCADE collector."""

import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class FLACCompressor:
    """FLAC compression handler for IQ data."""

    def __init__(self, compression_level: int = 5):
        """Initialize FLAC compressor.

        Args:
            compression_level: FLAC compression level (0-8)
        """
        self.compression_level = compression_level
        self.total_compressed = 0
        self.total_uncompressed = 0

    async def compress_file(self, input_path: str, output_path: str) -> Tuple[str, float]:
        """Compress WAV file to FLAC.

        Args:
            input_path: Input WAV file path
            output_path: Output FLAC file path

        Returns:
            Tuple of (output_path, compression_ratio)
        """
        input_p = Path(input_path)
        output_p = Path(output_path)

        # Read WAV file
        data, sample_rate = sf.read(str(input_p))

        # Get uncompressed size
        uncompressed_size = input_p.stat().st_size

        # Write FLAC file
        sf.write(
            str(output_p),
            data,
            sample_rate,
            subtype='PCM_16',
            format='FLAC'
        )

        # Get compressed size
        compressed_size = output_p.stat().st_size

        # Calculate compression ratio
        compression_ratio = 1 - (compressed_size / uncompressed_size)

        # Update totals
        self.total_uncompressed += uncompressed_size
        self.total_compressed += compressed_size

        logger.info(f"Compressed {input_p.name}: "
                   f"{uncompressed_size / 1024:.1f}KB -> {compressed_size / 1024:.1f}KB "
                   f"(ratio: {compression_ratio:.1%})")

        return str(output_p), compression_ratio

    def compress_iq_data(self, iq_data: np.ndarray, sample_rate: int,
                        output_path: str) -> Tuple[str, float]:
        """Compress IQ data directly to FLAC.

        Args:
            iq_data: IQ samples array
            sample_rate: Sample rate in Hz
            output_path: Output file path

        Returns:
            Tuple of (output_path, compression_ratio)
        """
        output_p = Path(output_path)

        # Prepare IQ data
        if np.iscomplexobj(iq_data):
            # Convert complex to 2-channel real
            iq_real = np.column_stack([iq_data.real, iq_data.imag])
        elif len(iq_data.shape) == 1:
            # Assume alternating I/Q samples
            iq_real = iq_data.reshape(-1, 2)
        else:
            iq_real = iq_data

        # Normalize to 16-bit range if needed
        if iq_real.dtype in [np.float32, np.float64]:
            max_val = np.abs(iq_real).max()
            if max_val > 0:
                iq_real = iq_real / max_val
            iq_int16 = (iq_real * 32767).astype(np.int16)
        else:
            iq_int16 = iq_real.astype(np.int16)

        # Calculate uncompressed size
        uncompressed_size = iq_int16.nbytes

        # Write FLAC file
        sf.write(
            str(output_p),
            iq_int16,
            sample_rate,
            subtype='PCM_16',
            format='FLAC'
        )

        # Calculate compression ratio
        compressed_size = output_p.stat().st_size
        compression_ratio = 1 - (compressed_size / uncompressed_size)

        logger.info(f"Compressed IQ data: "
                   f"{uncompressed_size / 1024 / 1024:.1f}MB -> "
                   f"{compressed_size / 1024 / 1024:.1f}MB "
                   f"(ratio: {compression_ratio:.1%})")

        return str(output_p), compression_ratio

    def get_statistics(self) -> dict:
        """Get compression statistics.

        Returns:
            Dictionary with compression stats
        """
        if self.total_uncompressed > 0:
            overall_ratio = 1 - (self.total_compressed / self.total_uncompressed)
        else:
            overall_ratio = 0

        return {
            "total_uncompressed_mb": self.total_uncompressed / 1024 / 1024,
            "total_compressed_mb": self.total_compressed / 1024 / 1024,
            "overall_compression_ratio": overall_ratio,
            "space_saved_mb": (self.total_uncompressed - self.total_compressed) / 1024 / 1024
        }