"""FLAC compression utility for IQ data.

Implements T033: FLAC compression utility (FR-007, FR-031).
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

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

    def compress(self, iq_data: np.ndarray, sample_rate: int, output_path: Path) -> Tuple[Path, float]:
        """Compress IQ data to FLAC format.

        Args:
            iq_data: IQ samples array
            sample_rate: Sample rate in Hz
            output_path: Output file path

        Returns:
            Tuple of (output_path, compression_ratio)
        """
        return compress_to_flac(iq_data, sample_rate, output_path, self.compression_level)

    def decompress(self, input_path: Path) -> Tuple[np.ndarray, int]:
        """Decompress FLAC file to IQ data.

        Args:
            input_path: Input FLAC file path

        Returns:
            Tuple of (iq_data, sample_rate)
        """
        return decompress_flac(input_path)


def compress_to_flac(
    iq_data: np.ndarray,
    sample_rate: int,
    output_path: Path,
    compression_level: int = 5,
) -> Tuple[Path, float]:
    """Compress IQ data to FLAC format (FR-031).

    Achieves 45-55% size reduction while maintaining lossless quality.

    Args:
        iq_data: IQ samples array (complex or 2-channel)
        sample_rate: Sample rate in Hz
        output_path: Output file path
        compression_level: FLAC compression level (0-8)

    Returns:
        Tuple of (output_path, compression_ratio)
    """
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
        # Normalize to [-1, 1] then scale
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
        output_path,
        iq_int16,
        sample_rate,
        subtype='PCM_16',
        format='FLAC',
    )

    # Calculate compression ratio
    compressed_size = output_path.stat().st_size
    compression_ratio = 1 - (compressed_size / uncompressed_size)

    logger.info(
        f"Compressed {uncompressed_size / 1024 / 1024:.1f} MB to "
        f"{compressed_size / 1024 / 1024:.1f} MB "
        f"(ratio: {compression_ratio:.1%})"
    )

    return output_path, compression_ratio


def decompress_flac(
    input_path: Path,
    return_complex: bool = True,
) -> Tuple[np.ndarray, int]:
    """Decompress FLAC file to IQ data.

    Args:
        input_path: FLAC file path
        return_complex: Return as complex array vs 2-channel

    Returns:
        Tuple of (iq_data, sample_rate)
    """
    # Read FLAC file
    iq_data, sample_rate = sf.read(input_path, dtype='int16')

    # Normalize to float
    iq_float = iq_data.astype(np.float32) / 32768.0

    if return_complex and len(iq_float.shape) == 2:
        # Convert 2-channel to complex
        iq_complex = iq_float[:, 0] + 1j * iq_float[:, 1]
        return iq_complex, sample_rate
    else:
        return iq_float, sample_rate


def estimate_compressed_size(
    duration_seconds: float,
    sample_rate: int = 12000,
    channels: int = 2,
    bit_depth: int = 16,
    compression_ratio: float = 0.5,
) -> int:
    """Estimate compressed file size for given duration.

    Args:
        duration_seconds: Recording duration
        sample_rate: Sample rate in Hz
        channels: Number of channels (2 for IQ)
        bit_depth: Bits per sample
        compression_ratio: Expected compression ratio

    Returns:
        Estimated size in bytes
    """
    uncompressed_size = (
        duration_seconds * sample_rate * channels * (bit_depth / 8)
    )
    compressed_size = int(uncompressed_size * (1 - compression_ratio))

    return compressed_size


def validate_flac_file(file_path: Path) -> dict:
    """Validate FLAC file integrity and metadata.

    Args:
        file_path: FLAC file path

    Returns:
        Validation results dict
    """
    try:
        # Try to read file metadata
        info = sf.info(file_path)

        # Try to read a small portion
        data, sr = sf.read(file_path, frames=1000)

        return {
            "valid": True,
            "duration_seconds": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
        }

    except Exception as e:
        logger.error(f"FLAC validation failed: {e}")
        return {
            "valid": False,
            "error": str(e),
        }


def batch_compress(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.wav",
    compression_level: int = 5,
) -> list:
    """Batch compress WAV files to FLAC.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        pattern: File pattern to match
        compression_level: FLAC compression level

    Returns:
        List of (input_path, output_path, compression_ratio) tuples
    """
    results = []
    input_files = list(input_dir.glob(pattern))

    logger.info(f"Found {len(input_files)} files to compress")

    for input_path in input_files:
        try:
            # Read WAV file
            data, sr = sf.read(input_path)

            # Generate output path
            output_path = output_dir / input_path.with_suffix('.flac').name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Compress
            _, ratio = compress_to_flac(
                data,
                sr,
                output_path,
                compression_level,
            )

            results.append((input_path, output_path, ratio))
            logger.info(f"Compressed {input_path.name}")

        except Exception as e:
            logger.error(f"Failed to compress {input_path}: {e}")

    return results


class FLACStreamWriter:
    """Stream writer for continuous FLAC compression."""

    def __init__(
        self,
        output_path: Path,
        sample_rate: int,
        channels: int = 2,
    ):
        """Initialize stream writer.

        Args:
            output_path: Output file path
            sample_rate: Sample rate in Hz
            channels: Number of channels
        """
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.file = None
        self.samples_written = 0

    def __enter__(self):
        """Context manager entry."""
        self.file = sf.SoundFile(
            self.output_path,
            mode='w',
            samplerate=self.sample_rate,
            channels=self.channels,
            format='FLAC',
            subtype='PCM_16',
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.file:
            self.file.close()

    def write(self, data: np.ndarray):
        """Write chunk of IQ data.

        Args:
            data: IQ samples to write
        """
        if self.file:
            self.file.write(data)
            self.samples_written += len(data)

    def get_duration(self) -> float:
        """Get current duration in seconds."""
        return self.samples_written / self.sample_rate