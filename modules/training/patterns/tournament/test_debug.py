#!/usr/bin/env python3
"""Test script to verify process pool functionality"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Add tournament to path
sys.path.insert(0, str(Path(__file__).parent))

from core.tournament_optimizer import simple_test_worker

def test_process_pool():
    """Test if process pool works with our simple worker"""
    print("Testing process pool...")

    try:
        with ProcessPoolExecutor(max_workers=2) as executor:
            # Submit some test work
            futures = []
            for i in range(4):
                future = executor.submit(simple_test_worker, i)
                futures.append(future)
                print(f"Submitted test job {i}")

            # Collect results
            results = []
            for future in futures:
                result = future.result(timeout=5)
                results.append(result)
                print(f"Got result: {result}")

            print(f"All results: {results}")
            print("Process pool test PASSED!")
            return True

    except Exception as e:
        print(f"Process pool test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_process_pool()
    sys.exit(0 if success else 1)