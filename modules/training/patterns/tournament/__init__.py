"""CASCADE Pattern Tournament Generator

Tournament-style pattern generation with dynamic compute allocation
and early stopping for optimal resource utilization.
"""

from .core.tournament_optimizer import DynamicTournamentOptimizer
from .core.trial_manager import Trial
from .core.elimination_strategy import EliminationStrategy, EliminationConfig
from .core.core_manager import CoreManager
from .ui.dashboard import PatternGeneratorDashboard
from .ui.logger import DualLogger

__version__ = "1.0.0"

__all__ = [
    'DynamicTournamentOptimizer',
    'Trial',
    'EliminationStrategy',
    'EliminationConfig',
    'CoreManager',
    'PatternGeneratorDashboard',
    'DualLogger'
]