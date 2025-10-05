"""Basic import tests for tournament module"""

import sys
from pathlib import Path

# Add both tournament and patterns directories to path
tournament_dir = Path(__file__).parent.parent
patterns_dir = tournament_dir.parent
sys.path.insert(0, str(patterns_dir))
sys.path.insert(0, str(tournament_dir))

def test_core_imports():
    """Test that core modules can be imported"""
    from tournament.core import (
        DynamicTournamentOptimizer,
        Trial,
        EliminationStrategy,
        CoreManager
    )
    assert DynamicTournamentOptimizer is not None
    assert Trial is not None
    assert EliminationStrategy is not None
    assert CoreManager is not None
    print("✓ Core imports successful")

def test_ui_imports():
    """Test that UI modules can be imported"""
    from tournament.ui import (
        PatternGeneratorDashboard,
        DualLogger
    )
    assert PatternGeneratorDashboard is not None
    assert DualLogger is not None
    print("✓ UI imports successful")

def test_main_import():
    """Test that main module can be imported"""
    import tournament
    assert tournament.__version__ == "1.0.0"
    print("✓ Main module import successful")

if __name__ == "__main__":
    test_core_imports()
    test_ui_imports()
    test_main_import()
    print("\n✓ All imports successful!")