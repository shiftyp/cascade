"""Waterfall generator for IQ samples (T051a).

Implements FR-044 for generating waterfall displays from IQ data.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, Any, Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WaterfallConfig:
    """Configuration for waterfall generation."""
    fft_size: int = 1024
    overlap: float = 0.5
    window: str = "hamming"
    colormap: str = "viridis"
    db_min: float = -120.0
    db_max: float = -40.0
    time_resolution: float = 0.1  # seconds per line
    frequency_resolution: float = 100.0  # Hz per bin


class WaterfallGenerator:
    """Generate waterfall displays from IQ samples (FR-044)."""

    def __init__(self, config: Optional[WaterfallConfig] = None):
        """Initialize waterfall generator.

        Args:
            config: Waterfall configuration
        """
        self.config = config or WaterfallConfig()
        self._window_cache = {}

    def generate(
        self,
        iq_data: np.ndarray,
        sample_rate: float,
        fft_size: Optional[int] = None,
        overlap: Optional[float] = None,
        colormap: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate waterfall data from IQ samples.

        Args:
            iq_data: Complex IQ samples
            sample_rate: Sample rate in Hz
            fft_size: FFT size (default from config)
            overlap: Overlap fraction 0-1 (default from config)
            colormap: Colormap name (default from config)

        Returns:
            Dictionary containing waterfall data and metadata
        """
        # Use provided parameters or defaults
        fft_size = fft_size or self.config.fft_size
        overlap = overlap or self.config.overlap
        colormap = colormap or self.config.colormap

        # Validate inputs
        if len(iq_data) < fft_size:
            logger.warning(f"IQ data length {len(iq_data)} < FFT size {fft_size}, padding")
            iq_data = np.pad(iq_data, (0, fft_size - len(iq_data)), 'constant')

        # Calculate spectrogram
        f, t, Sxx = self._compute_spectrogram(
            iq_data,
            sample_rate,
            fft_size,
            overlap
        )

        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Apply dynamic range limits
        Sxx_db = np.clip(Sxx_db, self.config.db_min, self.config.db_max)

        # Normalize for display
        Sxx_normalized = (Sxx_db - self.config.db_min) / (self.config.db_max - self.config.db_min)

        # Apply colormap
        waterfall_image = self._apply_colormap(Sxx_normalized, colormap)

        # Calculate statistics
        stats = self._calculate_statistics(Sxx_db, f, t)

        return {
            "data": Sxx_normalized,
            "data_db": Sxx_db,
            "image": waterfall_image,
            "frequencies": f,
            "timestamps": t,
            "sample_rate": sample_rate,
            "fft_size": fft_size,
            "overlap": overlap,
            "colormap": colormap,
            "statistics": stats,
            "config": {
                "db_min": self.config.db_min,
                "db_max": self.config.db_max,
                "window": self.config.window
            }
        }

    def _compute_spectrogram(
        self,
        iq_data: np.ndarray,
        sample_rate: float,
        fft_size: int,
        overlap: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram using STFT.

        Args:
            iq_data: Complex IQ samples
            sample_rate: Sample rate
            fft_size: FFT size
            overlap: Overlap fraction

        Returns:
            Tuple of (frequencies, times, spectrogram)
        """
        # Get window function
        window = self._get_window(fft_size)

        # Calculate hop size
        hop_size = int(fft_size * (1 - overlap))

        # Number of time bins
        n_bins = (len(iq_data) - fft_size) // hop_size + 1

        # Initialize spectrogram
        Sxx = np.zeros((fft_size, n_bins), dtype=np.float32)

        # Compute STFT
        for i in range(n_bins):
            start_idx = i * hop_size
            end_idx = start_idx + fft_size

            # Extract segment
            segment = iq_data[start_idx:end_idx]

            # Apply window
            windowed = segment * window

            # Compute FFT
            fft_result = fft(windowed, n=fft_size)

            # Shift zero frequency to center
            fft_shifted = np.fft.fftshift(fft_result)

            # Compute power spectral density
            Sxx[:, i] = np.abs(fft_shifted) ** 2

        # Generate frequency axis (centered at 0)
        frequencies = np.fft.fftshift(fftfreq(fft_size, 1/sample_rate))

        # Generate time axis
        times = np.arange(n_bins) * hop_size / sample_rate

        return frequencies, times, Sxx

    def _get_window(self, fft_size: int) -> np.ndarray:
        """Get window function (cached).

        Args:
            fft_size: Window size

        Returns:
            Window function
        """
        cache_key = (self.config.window, fft_size)

        if cache_key not in self._window_cache:
            if self.config.window == "hamming":
                window = np.hamming(fft_size)
            elif self.config.window == "hanning":
                window = np.hanning(fft_size)
            elif self.config.window == "blackman":
                window = np.blackman(fft_size)
            elif self.config.window == "bartlett":
                window = np.bartlett(fft_size)
            else:
                window = np.ones(fft_size)

            self._window_cache[cache_key] = window.astype(np.float32)

        return self._window_cache[cache_key]

    def _apply_colormap(
        self,
        data: np.ndarray,
        colormap_name: str
    ) -> np.ndarray:
        """Apply colormap to normalized data.

        Args:
            data: Normalized data (0-1)
            colormap_name: Name of matplotlib colormap

        Returns:
            RGB image array
        """
        try:
            cmap = plt.get_cmap(colormap_name)
        except ValueError:
            logger.warning(f"Unknown colormap {colormap_name}, using viridis")
            cmap = plt.get_cmap("viridis")

        # Apply colormap
        rgba = cmap(data)

        # Convert to RGB (drop alpha channel)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

        return rgb

    def _calculate_statistics(
        self,
        Sxx_db: np.ndarray,
        frequencies: np.ndarray,
        times: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate waterfall statistics.

        Args:
            Sxx_db: Spectrogram in dB
            frequencies: Frequency axis
            times: Time axis

        Returns:
            Statistics dictionary
        """
        return {
            "mean_power_db": float(np.mean(Sxx_db)),
            "max_power_db": float(np.max(Sxx_db)),
            "min_power_db": float(np.min(Sxx_db)),
            "std_power_db": float(np.std(Sxx_db)),
            "peak_frequency": float(frequencies[np.unravel_index(np.argmax(Sxx_db), Sxx_db.shape)[0]]),
            "peak_time": float(times[np.unravel_index(np.argmax(Sxx_db), Sxx_db.shape)[1]]) if len(times) > 0 else 0.0,
            "bandwidth": float(frequencies[-1] - frequencies[0]),
            "duration": float(times[-1] - times[0]) if len(times) > 1 else 0.0,
            "frequency_resolution": float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.0,
            "time_resolution": float(times[1] - times[0]) if len(times) > 1 else 0.0
        }

    def generate_thumbnail(
        self,
        iq_data: np.ndarray,
        sample_rate: float,
        width: int = 256,
        height: int = 128
    ) -> np.ndarray:
        """Generate thumbnail waterfall for quick preview.

        Args:
            iq_data: Complex IQ samples
            sample_rate: Sample rate
            width: Thumbnail width in pixels
            height: Thumbnail height in pixels

        Returns:
            RGB thumbnail image
        """
        # Use smaller FFT size for thumbnail
        fft_size = min(256, len(iq_data) // 4)

        # Generate waterfall
        waterfall = self.generate(
            iq_data,
            sample_rate,
            fft_size=fft_size,
            overlap=0.25  # Less overlap for speed
        )

        # Resize to thumbnail dimensions
        from scipy.ndimage import zoom

        data = waterfall["data"]
        zoom_factors = (height / data.shape[0], width / data.shape[1])
        thumbnail_data = zoom(data, zoom_factors, order=1)

        # Apply colormap
        thumbnail_image = self._apply_colormap(
            thumbnail_data,
            self.config.colormap
        )

        return thumbnail_image

    def generate_animated_waterfall(
        self,
        iq_stream_generator,
        sample_rate: float,
        history_size: int = 100
    ):
        """Generate animated waterfall for real-time display.

        Args:
            iq_stream_generator: Generator yielding IQ data chunks
            sample_rate: Sample rate
            history_size: Number of time slices to keep in history

        Yields:
            Waterfall updates for each new chunk
        """
        # Initialize history buffer
        history = []

        for iq_chunk in iq_stream_generator:
            # Generate waterfall for chunk
            waterfall = self.generate(
                iq_chunk,
                sample_rate,
                fft_size=self.config.fft_size,
                overlap=0
            )

            # Add to history
            history.append(waterfall["data"][:, 0])

            # Maintain history size
            if len(history) > history_size:
                history.pop(0)

            # Stack history into full waterfall
            full_waterfall = np.column_stack(history)

            yield {
                "waterfall": full_waterfall,
                "latest_column": waterfall["data"][:, 0],
                "frequencies": waterfall["frequencies"],
                "time_index": len(history)
            }

    def detect_signals(
        self,
        waterfall_data: np.ndarray,
        threshold_db: float = -90.0,
        min_bandwidth_hz: float = 100.0,
        min_duration_sec: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Detect signals in waterfall.

        Args:
            waterfall_data: Waterfall data in dB
            threshold_db: Detection threshold
            min_bandwidth_hz: Minimum signal bandwidth
            min_duration_sec: Minimum signal duration

        Returns:
            List of detected signals
        """
        # Binary mask of signals above threshold
        signal_mask = waterfall_data > threshold_db

        # Find connected components
        from scipy.ndimage import label, measurements

        labeled, num_features = label(signal_mask)

        signals = []
        for i in range(1, num_features + 1):
            # Get signal region
            signal_indices = np.where(labeled == i)

            if len(signal_indices[0]) == 0:
                continue

            # Calculate signal properties
            freq_start = signal_indices[0].min()
            freq_end = signal_indices[0].max()
            time_start = signal_indices[1].min()
            time_end = signal_indices[1].max()

            # Check minimum requirements
            bandwidth = freq_end - freq_start
            duration = time_end - time_start

            if bandwidth >= min_bandwidth_hz and duration >= min_duration_sec:
                # Calculate signal statistics
                signal_region = waterfall_data[signal_indices]

                signals.append({
                    "id": i,
                    "freq_start": freq_start,
                    "freq_end": freq_end,
                    "time_start": time_start,
                    "time_end": time_end,
                    "bandwidth": bandwidth,
                    "duration": duration,
                    "mean_power_db": float(np.mean(signal_region)),
                    "max_power_db": float(np.max(signal_region)),
                    "area": len(signal_indices[0])
                })

        return signals


# Utility functions
def generate_test_waterfall() -> Dict[str, Any]:
    """Generate test waterfall for development.

    Returns:
        Test waterfall data
    """
    # Generate test IQ data with signals
    sample_rate = 12000.0
    duration = 10.0
    t = np.arange(0, duration, 1/sample_rate)

    # Add multiple test signals
    iq_data = np.zeros(len(t), dtype=np.complex64)

    # CW signal at 1 kHz
    iq_data += 0.5 * np.exp(1j * 2 * np.pi * 1000 * t)

    # Chirp from -2 kHz to 2 kHz
    f_start = -2000
    f_end = 2000
    chirp_rate = (f_end - f_start) / duration
    iq_data += 0.3 * np.exp(1j * 2 * np.pi * (f_start * t + 0.5 * chirp_rate * t**2))

    # Add noise
    iq_data += 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))

    # Generate waterfall
    generator = WaterfallGenerator()
    waterfall = generator.generate(iq_data, sample_rate)

    return waterfall