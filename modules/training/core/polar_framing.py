"""
Message framing for efficient polar encoding with large messages.

Splits large messages into fixed-size frames to avoid polar code overhead.
Each frame is independently encoded with polar codes.
"""

import numpy as np
from typing import List, Tuple


def frame_message(message_bits: np.ndarray, frame_size_bits: int = 512) -> List[np.ndarray]:
    """
    Split message into fixed-size frames for efficient polar encoding.

    Args:
        message_bits: Message bits to frame
        frame_size_bits: Frame size in bits (default: 512)

    Returns:
        List of frames (each ≤ frame_size_bits)
    """
    num_bits = len(message_bits)
    num_frames = int(np.ceil(num_bits / frame_size_bits))

    frames = []
    for i in range(num_frames):
        start = i * frame_size_bits
        end = min((i + 1) * frame_size_bits, num_bits)
        frame = message_bits[start:end]
        frames.append(frame)

    return frames


def should_frame_message(message_bytes: bytes, threshold_bytes: int = 64) -> bool:
    """
    Determine if message should be framed based on size.

    Args:
        message_bytes: Message to check
        threshold_bytes: Frame if larger than this

    Returns:
        True if message should be framed
    """
    return len(message_bytes) > threshold_bytes


def calculate_optimal_frame_size(message_bits: int, fec_rate: Tuple[int, int] = (2, 3)) -> int:
    """
    Calculate optimal frame size to minimize overhead.

    Target: Frame should encode to 512 or 1024 bit polar blocks.

    Args:
        message_bits: Total message size
        fec_rate: FEC rate (k, n)

    Returns:
        Optimal frame size in bits
    """
    k, n = fec_rate

    # For 512-bit polar: Need ~340 bits input (with rate 2/3)
    # For 1024-bit polar: Need ~680 bits input

    if message_bits < 340:
        return message_bits  # Single frame

    # Use 512-bit frames (encode to 1024-bit polar blocks)
    # Each frame: 512 bits → 768 after FEC → 1024 polar
    return 512
