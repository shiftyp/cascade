"""IQ file streaming reader for FLAC samples (T051b).

Handles efficient streaming of IQ data from compressed FLAC files.
"""

import numpy as np
import soundfile as sf
import asyncio
from pathlib import Path
from typing import Optional, Tuple, AsyncGenerator, Dict, Any, List
import logging
import struct
from dataclasses import dataclass
import mmap
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class IQFileInfo:
    """Information about an IQ file."""
    file_path: Path
    sample_rate: int
    num_samples: int
    duration_seconds: float
    file_size_bytes: int
    format: str
    channels: int
    is_complex: bool


class IQStreamReader:
    """Stream IQ data from FLAC files efficiently."""

    def __init__(self, cache_size_mb: int = 100):
        """Initialize IQ stream reader.

        Args:
            cache_size_mb: Cache size in megabytes
        """
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self._file_cache = {}
        self._info_cache = {}

    async def get_file_info(self, file_path: str) -> IQFileInfo:
        """Get information about an IQ file.

        Args:
            file_path: Path to IQ file

        Returns:
            File information
        """
        path = Path(file_path)

        # Check cache
        if file_path in self._info_cache:
            return self._info_cache[file_path]

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Get file info using soundfile
            with sf.SoundFile(file_path) as f:
                info = IQFileInfo(
                    file_path=path,
                    sample_rate=f.samplerate,
                    num_samples=len(f),
                    duration_seconds=len(f) / f.samplerate,
                    file_size_bytes=path.stat().st_size,
                    format=f.format,
                    channels=f.channels,
                    is_complex=f.channels == 2  # I/Q as stereo channels
                )

            # Cache the info
            self._info_cache[file_path] = info

            return info

        except Exception as e:
            logger.error(f"Error reading file info from {file_path}: {e}")
            raise

    async def read_segment(
        self,
        file_path: str,
        start_time: float,
        duration: float,
        return_complex: bool = True
    ) -> np.ndarray:
        """Read a segment of IQ data from file.

        Args:
            file_path: Path to IQ file
            start_time: Start time in seconds
            duration: Duration in seconds
            return_complex: Return as complex array (True) or I/Q channels (False)

        Returns:
            IQ data as numpy array
        """
        info = await self.get_file_info(file_path)

        # Calculate sample indices
        start_sample = int(start_time * info.sample_rate)
        num_samples = int(duration * info.sample_rate)

        # Clip to file boundaries
        start_sample = max(0, min(start_sample, info.num_samples))
        num_samples = min(num_samples, info.num_samples - start_sample)

        if num_samples <= 0:
            return np.array([], dtype=np.complex64 if return_complex else np.float32)

        try:
            # Read from FLAC file
            data, _ = sf.read(
                file_path,
                start=start_sample,
                stop=start_sample + num_samples,
                dtype='float32'
            )

            # Convert to complex if needed
            if return_complex and info.channels == 2:
                # Assume I is channel 0, Q is channel 1
                iq_complex = data[:, 0] + 1j * data[:, 1]
                return iq_complex.astype(np.complex64)
            else:
                return data

        except Exception as e:
            logger.error(f"Error reading segment from {file_path}: {e}")
            raise

    async def stream_chunks(
        self,
        file_path: str,
        chunk_duration: float = 1.0,
        overlap: float = 0.0,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> AsyncGenerator[Tuple[np.ndarray, float], None]:
        """Stream file in chunks for real-time processing.

        Args:
            file_path: Path to IQ file
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks (0-1)
            start_time: Start time in seconds
            end_time: End time in seconds (None for entire file)

        Yields:
            Tuples of (IQ data chunk, timestamp)
        """
        info = await self.get_file_info(file_path)

        # Determine end time
        if end_time is None:
            end_time = info.duration_seconds
        else:
            end_time = min(end_time, info.duration_seconds)

        # Calculate hop size
        hop_duration = chunk_duration * (1 - overlap)

        current_time = start_time

        while current_time < end_time:
            # Read chunk
            chunk = await self.read_segment(
                file_path,
                current_time,
                min(chunk_duration, end_time - current_time)
            )

            if len(chunk) == 0:
                break

            yield chunk, current_time

            current_time += hop_duration

            # Small delay to prevent blocking
            await asyncio.sleep(0.001)

    async def read_decimated(
        self,
        file_path: str,
        decimation_factor: int,
        start_time: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """Read decimated IQ data for overview display.

        Args:
            file_path: Path to IQ file
            decimation_factor: Decimation factor
            start_time: Start time in seconds
            duration: Duration in seconds (None for entire file)

        Returns:
            Tuple of (decimated IQ data, new sample rate)
        """
        info = await self.get_file_info(file_path)

        # Read full resolution data
        if duration is None:
            duration = info.duration_seconds - start_time

        data = await self.read_segment(file_path, start_time, duration)

        # Decimate
        if decimation_factor > 1:
            # Apply anti-aliasing filter before decimation
            from scipy import signal

            # Design lowpass filter
            nyquist = 0.5 * info.sample_rate
            cutoff = nyquist / decimation_factor * 0.8  # 80% of Nyquist
            sos = signal.butter(8, cutoff, btype='low', fs=info.sample_rate, output='sos')

            # Filter and decimate
            filtered = signal.sosfilt(sos, data)
            decimated = filtered[::decimation_factor]

            new_sample_rate = info.sample_rate // decimation_factor
        else:
            decimated = data
            new_sample_rate = info.sample_rate

        return decimated, new_sample_rate

    async def extract_samples_for_training(
        self,
        file_path: str,
        sample_duration: float = 0.1,
        num_samples: int = 100,
        random_selection: bool = True
    ) -> List[np.ndarray]:
        """Extract samples for model training.

        Args:
            file_path: Path to IQ file
            sample_duration: Duration of each sample in seconds
            num_samples: Number of samples to extract
            random_selection: Random (True) or evenly spaced (False)

        Returns:
            List of IQ samples
        """
        info = await self.get_file_info(file_path)

        samples = []

        if random_selection:
            # Random sampling
            import random

            for _ in range(num_samples):
                # Random start time
                max_start = info.duration_seconds - sample_duration
                if max_start <= 0:
                    break

                start_time = random.uniform(0, max_start)

                # Read sample
                sample = await self.read_segment(
                    file_path,
                    start_time,
                    sample_duration
                )

                samples.append(sample)
        else:
            # Evenly spaced sampling
            interval = (info.duration_seconds - sample_duration) / (num_samples - 1)

            for i in range(num_samples):
                start_time = i * interval

                if start_time + sample_duration > info.duration_seconds:
                    break

                # Read sample
                sample = await self.read_segment(
                    file_path,
                    start_time,
                    sample_duration
                )

                samples.append(sample)

        return samples

    async def convert_to_baseband(
        self,
        file_path: str,
        center_frequency_hz: float,
        target_frequency_hz: float = 0.0
    ) -> AsyncGenerator[np.ndarray, None]:
        """Convert IF signal to baseband.

        Args:
            file_path: Path to IQ file
            center_frequency_hz: Center frequency of recording
            target_frequency_hz: Target frequency to shift to baseband

        Yields:
            Baseband IQ chunks
        """
        info = await self.get_file_info(file_path)

        # Calculate frequency shift
        freq_shift = target_frequency_hz - center_frequency_hz

        # Create complex exponential for mixing
        chunk_size = info.sample_rate  # 1 second chunks
        t = np.arange(chunk_size) / info.sample_rate
        mixer = np.exp(-1j * 2 * np.pi * freq_shift * t).astype(np.complex64)

        async for chunk, timestamp in self.stream_chunks(file_path, chunk_duration=1.0):
            # Apply frequency shift
            if len(chunk) == len(mixer):
                baseband = chunk * mixer
            else:
                # Handle last chunk
                mixer_short = mixer[:len(chunk)]
                baseband = chunk * mixer_short

            yield baseband

    def clear_cache(self):
        """Clear file and info caches."""
        self._file_cache.clear()
        self._info_cache.clear()
        logger.info("IQ reader cache cleared")


class FLACMetadataReader:
    """Read metadata from FLAC files."""

    @staticmethod
    def read_metadata(file_path: str) -> Dict[str, Any]:
        """Read CASCADE-specific metadata from FLAC file.

        Args:
            file_path: Path to FLAC file

        Returns:
            Metadata dictionary
        """
        try:
            with sf.SoundFile(file_path) as f:
                # Standard metadata
                metadata = {
                    "sample_rate": f.samplerate,
                    "channels": f.channels,
                    "frames": len(f),
                    "duration": len(f) / f.samplerate,
                    "format": f.format,
                    "subtype": f.subtype
                }

                # Try to read custom metadata if available
                if hasattr(f, 'extra_info'):
                    extra = f.extra_info

                    # CASCADE-specific metadata
                    metadata.update({
                        "frequency_khz": extra.get("frequency_khz"),
                        "band": extra.get("band"),
                        "grid_square": extra.get("grid_square"),
                        "correlation_id": extra.get("correlation_id"),
                        "session_id": extra.get("session_id"),
                        "timestamp": extra.get("timestamp"),
                        "propagation_mode": extra.get("propagation_mode"),
                        "signal_type": extra.get("signal_type")
                    })

                return metadata

        except Exception as e:
            logger.error(f"Error reading metadata from {file_path}: {e}")
            return {}


# Utility functions for testing
async def test_iq_reader():
    """Test IQ reader functionality."""
    reader = IQStreamReader()

    # Generate test FLAC file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as tmp:
        # Generate test IQ data
        sample_rate = 12000
        duration = 5.0
        t = np.arange(0, duration, 1/sample_rate)

        # Create test signal (1 kHz tone)
        i_channel = np.cos(2 * np.pi * 1000 * t)
        q_channel = np.sin(2 * np.pi * 1000 * t)

        # Stack as stereo
        iq_stereo = np.column_stack((i_channel, q_channel))

        # Write to FLAC
        sf.write(tmp.name, iq_stereo, sample_rate, format='FLAC')

        # Test reading
        logger.info("Testing IQ reader...")

        # Get file info
        info = await reader.get_file_info(tmp.name)
        logger.info(f"File info: {info}")

        # Read segment
        segment = await reader.read_segment(tmp.name, 1.0, 0.5)
        logger.info(f"Segment shape: {segment.shape}, dtype: {segment.dtype}")

        # Stream chunks
        chunk_count = 0
        async for chunk, timestamp in reader.stream_chunks(tmp.name, chunk_duration=0.5):
            chunk_count += 1
            logger.info(f"Chunk {chunk_count}: shape={chunk.shape}, time={timestamp:.2f}s")
            if chunk_count >= 5:
                break

        logger.info("IQ reader test complete")

        return tmp.name


if __name__ == "__main__":
    # Run test
    asyncio.run(test_iq_reader())