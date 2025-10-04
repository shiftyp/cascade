"""Platform Detection and CPU Architecture Optimization

Auto-detects CPU capabilities and optimizes pattern generation accordingly.
"""

import os
import platform
from typing import Dict, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import cpuinfo
    HAS_CPUINFO = True
except ImportError:
    HAS_CPUINFO = False


def detect_optimal_workers() -> int:
    """Detect optimal number of parallel workers for this CPU

    Returns physical cores - 2 (reserve for OS/background tasks)

    Returns:
        Optimal worker count (minimum 1)
    """
    if HAS_PSUTIL:
        # Prefer physical cores over logical (avoid hyperthreading overhead)
        physical_cores = psutil.cpu_count(logical=False)
        if physical_cores:
            return max(1, physical_cores - 2)

    # Fallback to os.cpu_count()
    total_cores = os.cpu_count() or 4
    # Assume hyperthreading, divide by 2
    estimated_physical = max(1, total_cores // 2)
    return max(1, estimated_physical - 2)


def detect_hybrid_architecture() -> Dict[str, any]:
    """Detect Intel hybrid architecture (P-cores + E-cores)

    Returns:
        Dict with hybrid architecture info
    """
    result = {
        'has_hybrid': False,
        'p_cores': 0,
        'e_cores': 0,
        'strategy': 'none'
    }

    if not HAS_CPUINFO:
        return result

    try:
        cpu_info = cpuinfo.get_cpu_info()
        brand = cpu_info.get('brand_raw', '')

        # Detect Intel hybrid CPUs (12th gen+, Core Ultra)
        if 'Intel' in brand and ('Core Ultra' in brand or '12th' in brand or '13th' in brand or '14th' in brand):
            result['has_hybrid'] = True

            # Core Ultra 7 265K: 8P + 12E
            if 'Core Ultra' in brand:
                result['p_cores'] = 8
                result['e_cores'] = 12
                result['strategy'] = 'pin_to_p_cores'
            # 12th-14th gen varies
            else:
                total = psutil.cpu_count(logical=False) if HAS_PSUTIL else os.cpu_count()
                # Rough estimate: 40-60% are P-cores
                result['p_cores'] = int(total * 0.5)
                result['e_cores'] = total - result['p_cores']
                result['strategy'] = 'pin_to_p_cores'

    except Exception as e:
        pass  # Return default

    return result


def detect_memory_constraints() -> Dict[str, any]:
    """Detect available memory and suggest batch sizing

    Returns:
        Dict with memory info and recommendations
    """
    result = {
        'total_gb': 8.0,  # Default assumption
        'available_gb': 4.0,
        'max_parallel_trials': 8,
        'suggested_iterations': 100000
    }

    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        result['total_gb'] = mem.total / (1024**3)
        result['available_gb'] = mem.available / (1024**3)

        # Each trial needs ~500 MB, leave 4 GB for OS
        result['max_parallel_trials'] = max(1, int((result['available_gb'] - 4) / 0.5))

        # Reduce iterations on low-memory systems
        if result['total_gb'] < 8:
            result['suggested_iterations'] = 50000
        else:
            result['suggested_iterations'] = 100000

    return result


def detect_simd_capabilities() -> Dict[str, bool]:
    """Detect SIMD instruction set support

    Returns:
        Dict with SIMD capability flags
    """
    capabilities = {
        'avx512': False,
        'avx2': False,
        'sse4': False,
        'neon': False  # ARM
    }

    if HAS_CPUINFO:
        try:
            cpu_info = cpuinfo.get_cpu_info()
            flags = cpu_info.get('flags', [])

            capabilities['avx512'] = any('avx512' in f for f in flags)
            capabilities['avx2'] = 'avx2' in flags
            capabilities['sse4' in flags] = 'sse4_1' in flags or 'sse4_2' in flags

            # ARM NEON
            if 'neon' in flags or 'asimd' in flags:
                capabilities['neon'] = True

        except Exception:
            pass

    return capabilities


def get_platform_config() -> Dict[str, any]:
    """Get comprehensive platform configuration

    Returns:
        Dict with all platform detection results
    """
    config = {
        'os': platform.system(),
        'arch': platform.machine(),
        'python_version': platform.python_version(),
        'optimal_workers': detect_optimal_workers(),
        'hybrid': detect_hybrid_architecture(),
        'memory': detect_memory_constraints(),
        'simd': detect_simd_capabilities(),
        'has_psutil': HAS_PSUTIL,
        'has_cpuinfo': HAS_CPUINFO
    }

    return config


def optimize_for_architecture(config: Optional[Dict] = None) -> Dict[str, any]:
    """Apply platform-specific optimizations

    Args:
        config: Platform config (auto-detect if None)

    Returns:
        Dict with optimization settings applied
    """
    if config is None:
        config = get_platform_config()

    settings = {
        'num_workers': config['optimal_workers'],
        'max_iterations': config['memory']['suggested_iterations'],
        'use_p_cores_only': False,
        'numpy_threads': config['optimal_workers']
    }

    # Hybrid CPU: Pin to P-cores
    if config['hybrid']['has_hybrid'] and config['hybrid']['p_cores'] > 0:
        settings['use_p_cores_only'] = True
        settings['num_workers'] = config['hybrid']['p_cores']
        settings['p_core_list'] = list(range(config['hybrid']['p_cores']))

    # Set NumPy thread count
    os.environ['OMP_NUM_THREADS'] = str(settings['numpy_threads'])
    os.environ['MKL_NUM_THREADS'] = str(settings['numpy_threads'])
    os.environ['OPENBLAS_NUM_THREADS'] = str(settings['numpy_threads'])

    # Linux: Can set CPU affinity
    if config['os'] == 'Linux' and settings['use_p_cores_only']:
        try:
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, settings['p_core_list'])
                settings['affinity_set'] = True
        except Exception:
            settings['affinity_set'] = False

    return settings


def print_platform_info(config: Optional[Dict] = None):
    """Print platform detection results

    Args:
        config: Platform config (auto-detect if None)
    """
    if config is None:
        config = get_platform_config()

    print("=== Platform Detection ===")
    print(f"OS: {config['os']} ({config['arch']})")
    print(f"Python: {config['python_version']}")
    print(f"Optimal workers: {config['optimal_workers']}")

    if config['hybrid']['has_hybrid']:
        print(f"Hybrid CPU: {config['hybrid']['p_cores']} P-cores + {config['hybrid']['e_cores']} E-cores")
        print(f"Strategy: {config['hybrid']['strategy']}")

    print(f"Total RAM: {config['memory']['total_gb']:.1f} GB")
    print(f"Available RAM: {config['memory']['available_gb']:.1f} GB")
    print(f"Max parallel trials: {config['memory']['max_parallel_trials']}")
    print(f"Suggested iterations: {config['memory']['suggested_iterations']:,}")

    simd = config['simd']
    simd_list = [k.upper() for k, v in simd.items() if v]
    if simd_list:
        print(f"SIMD support: {', '.join(simd_list)}")

    if not config['has_psutil']:
        print("⚠ psutil not installed - using fallback detection")
    if not config['has_cpuinfo']:
        print("⚠ py-cpuinfo not installed - limited CPU detection")

    print()
