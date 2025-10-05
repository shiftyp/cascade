"""CASCADE Modem Configuration"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class CASCADEConfig(BaseSettings):
    """Configuration for CASCADE modem server"""

    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")

    # User Profile
    callsign: str = Field(default="NOCALL", description="Station callsign")
    grid_square: str = Field(default="AA00aa", description="Maidenhead grid square")
    hardware_tier: str = Field(
        default="rpi4",
        description="Hardware tier: rpi4, coral, desktop, gpu"
    )

    # Radio Settings (Hamlib)
    radio_model: Optional[int] = Field(default=None, description="Hamlib RIG_MODEL_* number")
    radio_port: str = Field(default="/dev/ttyUSB0", description="Radio serial port")
    radio_baud: int = Field(default=9600, description="CAT baud rate")

    # Frequency & Mode
    frequency: int = Field(default=14074000, description="Operating frequency (Hz)")
    mode: str = Field(default="USB", description="Operating mode")
    bandwidth: int = Field(default=3000, description="Filter bandwidth (Hz)")

    # Audio Settings
    audio_input_device: Optional[int] = Field(default=None, description="Input device index")
    audio_output_device: Optional[int] = Field(default=None, description="Output device index")
    sample_rate: int = Field(default=12000, description="Audio sample rate (Hz)")

    # CASCADE Protocol
    max_simultaneous_users: int = Field(
        default=15,
        description="Max simultaneous decode capacity (RPi4: 15, Desktop: 30-50)"
    )
    pattern_pool_size: int = Field(default=8, description="Assigned pattern pool size")

    # Model Settings
    model_path: Optional[str] = Field(
        default=None,
        description="Path to trained CASCADE PyTorch model"
    )

    # Telemetry (optional)
    telemetry_enabled: bool = Field(default=False, description="Enable telemetry upload")
    telemetry_endpoint: Optional[str] = Field(
        default=None,
        description="Telemetry API endpoint"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global config instance
config = CASCADEConfig()
