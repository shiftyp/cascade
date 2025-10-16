"""CASCADE V2 Channel Simulator Module.

Provides channel effects for training data generation:
- AWGN noise
- QRN (atmospheric noise)
- Multipath fading
- QRM (interference)
- Collision scenarios
- Channel orchestration
"""

from .awgn import generate_awgn
from .qrn import (
    generate_mixed_qrn,
    generate_continuous_static,
    generate_crackling_noise,
    generate_lightning_crashes,
    generate_powerline_noise
)
from .multipath import (
    apply_multipath_fading,
    MultipathProfile,
    watterson_hf_profile,
    severe_multipath_profile
)
from .qrm import generate_mixed_qrm
from .collisions import create_collision_scenarios, CollisionSignal
from .orchestrator import ChannelOrchestrator, ExpertConfig, get_expert_configs

__all__ = [
    # AWGN
    'generate_awgn',

    # QRN
    'generate_mixed_qrn',
    'generate_continuous_static',
    'generate_crackling_noise',
    'generate_lightning_crashes',
    'generate_powerline_noise',

    # Multipath
    'apply_multipath_fading',
    'MultipathProfile',
    'watterson_hf_profile',
    'severe_multipath_profile',

    # QRM
    'generate_mixed_qrm',

    # Collisions
    'create_collision_scenarios',
    'CollisionSignal',

    # Orchestrator
    'ChannelOrchestrator',
    'ExpertConfig',
    'get_expert_configs',
]
