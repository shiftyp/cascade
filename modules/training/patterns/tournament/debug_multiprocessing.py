#!/usr/bin/env python3
"""Debug script to diagnose multiprocessing access denied errors"""

import sys
import os
import traceback
from pathlib import Path
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def simple_worker(x):
    """Simple worker function for testing"""
    return x * 2

def test_basic_multiprocessing():
    """Test basic multiprocessing functionality"""
    print("Testing basic multiprocessing...")
    try:
        # Test 1: Basic multiprocessing
        with multiprocessing.Pool(2) as pool:
            result = pool.map(simple_worker, [1, 2, 3])
            print(f"  ✓ Basic multiprocessing works: {result}")
            return True
    except Exception as e:
        print(f"  ✗ Basic multiprocessing failed: {e}")
        traceback.print_exc()
        return False

def test_process_pool_executor():
    """Test ProcessPoolExecutor"""
    print("\nTesting ProcessPoolExecutor...")
    try:
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(simple_worker, i) for i in [1, 2, 3]]
            results = [f.result() for f in futures]
            print(f"  ✓ ProcessPoolExecutor works: {results}")
            return True
    except Exception as e:
        print(f"  ✗ ProcessPoolExecutor failed: {e}")
        traceback.print_exc()
        return False

def test_thread_pool_executor():
    """Test ThreadPoolExecutor"""
    print("\nTesting ThreadPoolExecutor...")
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(simple_worker, i) for i in [1, 2, 3]]
            results = [f.result() for f in futures]
            print(f"  ✓ ThreadPoolExecutor works: {results}")
            return True
    except Exception as e:
        print(f"  ✗ ThreadPoolExecutor failed: {e}")
        return False

def test_temp_directory_access():
    """Test if we can access temp directory (used by multiprocessing)"""
    print("\nTesting temp directory access...")
    try:
        temp_dir = tempfile.gettempdir()
        print(f"  Temp directory: {temp_dir}")

        # Try to create a file
        test_file = Path(temp_dir) / "multiprocessing_test.txt"
        test_file.write_text("test")
        test_file.unlink()
        print(f"  ✓ Can write to temp directory")
        return True
    except Exception as e:
        print(f"  ✗ Cannot write to temp directory: {e}")
        return False

def test_spawn_vs_fork():
    """Test different multiprocessing start methods"""
    print("\nTesting multiprocessing start methods...")
    current_method = multiprocessing.get_start_method()
    print(f"  Current start method: {current_method}")

    # On Windows, only 'spawn' is available
    # On Linux/Mac, 'fork', 'spawn', and 'forkserver' are available
    if sys.platform == "win32":
        print("  Windows detected - only 'spawn' method available")
        methods_to_test = ['spawn']
    else:
        methods_to_test = ['spawn', 'fork']

    for method in methods_to_test:
        try:
            ctx = multiprocessing.get_context(method)
            with ctx.Pool(2) as pool:
                result = pool.map(simple_worker, [1, 2])
                print(f"  ✓ Method '{method}' works: {result}")
        except Exception as e:
            print(f"  ✗ Method '{method}' failed: {e}")

def check_environment():
    """Check environment variables and system settings"""
    print("\nEnvironment Check:")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  OS: {os.name}")
    print(f"  Working directory: {os.getcwd()}")
    print(f"  User: {os.environ.get('USER', os.environ.get('USERNAME', 'unknown'))}")

    # Check if running as admin (Windows)
    if sys.platform == "win32":
        import ctypes
        try:
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            print(f"  Running as Administrator: {is_admin}")
        except:
            print(f"  Could not check admin status")

    # Check multiprocessing settings
    print(f"\nMultiprocessing settings:")
    print(f"  CPU count: {multiprocessing.cpu_count()}")
    print(f"  Start method: {multiprocessing.get_start_method()}")

def suggest_fixes():
    """Suggest potential fixes based on test results"""
    print("\n" + "=" * 60)
    print("SUGGESTED FIXES:")
    print("=" * 60)

    if sys.platform == "win32":
        print("\nFor Windows:")
        print("1. Try running with: --execution-mode thread")
        print("   This uses ThreadPoolExecutor which doesn't spawn new processes")
        print("")
        print("2. Check Windows Defender/Antivirus:")
        print("   - Add python.exe to exclusions")
        print("   - Temporarily disable real-time protection to test")
        print("")
        print("3. Check if running from OneDrive/network drive:")
        print("   - Copy to local drive (C:\\temp\\) and try again")
        print("")
        print("4. Try setting environment variable:")
        print("   set PYTHONOPTIMIZE=1")
        print("   python generate_patterns_tournament.py")
    else:
        print("\nFor Linux/Mac:")
        print("1. Try different execution modes:")
        print("   --execution-mode thread  (uses threads instead of processes)")
        print("   --execution-mode sequential  (no parallelism, but always works)")
        print("")
        print("2. Check ulimits:")
        print("   ulimit -n  (file descriptors)")
        print("   ulimit -u  (max processes)")
        print("")
        print("3. Check /tmp permissions:")
        print("   ls -la /tmp | head")

    print("\n" + "=" * 60)
    print("QUICK WORKAROUND:")
    print("=" * 60)
    print("\nRun with ThreadPoolExecutor (still parallel but uses threads):")
    print("python generate_patterns_tournament.py --execution-mode thread")
    print("\nThis provides parallelism without the multiprocessing overhead.")

if __name__ == "__main__":
    print("=" * 60)
    print("Multiprocessing Debug Tool")
    print("=" * 60)

    check_environment()

    all_tests_passed = True
    all_tests_passed &= test_temp_directory_access()
    all_tests_passed &= test_basic_multiprocessing()
    all_tests_passed &= test_process_pool_executor()
    all_tests_passed &= test_thread_pool_executor()
    test_spawn_vs_fork()

    suggest_fixes()

    if not all_tests_passed:
        print("\n⚠️  Some tests failed. See suggestions above.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! Multiprocessing should work.")
        sys.exit(0)