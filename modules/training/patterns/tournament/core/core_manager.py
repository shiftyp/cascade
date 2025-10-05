"""CPU core management for optimal P-core utilization

Manages CPU affinity, core rotation, and performance optimization
for Intel Core Ultra hybrid architectures.
"""

import os
import platform
import threading
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path


class CoreManager:
    """Manages CPU core assignment and optimization"""

    def __init__(self, num_p_cores: int = 8, rotate_cores: bool = True):
        self.num_p_cores = num_p_cores
        self.rotate_cores = rotate_cores
        self.rotation_interval = timedelta(minutes=30)
        self.last_rotation = datetime.now()

        # Detect system configuration
        self.platform = platform.system()
        self.is_windows = self.platform == "Windows"
        self.is_linux = self.platform == "Linux"
        self.is_mac = self.platform == "Darwin"

        # Core assignments
        self.p_core_ids = self._detect_p_cores()
        self.e_core_ids = self._detect_e_cores()
        self.trial_assignments: Dict[int, List[int]] = {}

        # Rotation tracking
        self.rotation_count = 0
        self.rotation_lock = threading.Lock()

    def _detect_p_cores(self) -> List[int]:
        """Detect P-cores on the system

        For Intel Core Ultra 7 265K:
        - P-cores are typically cores 0-7
        - E-cores are cores 8-19
        """
        if self.is_linux:
            # Try to read from sysfs
            p_cores = []
            cpu_path = Path("/sys/devices/system/cpu")

            for cpu_dir in sorted(cpu_path.glob("cpu[0-9]*")):
                cpu_num = int(cpu_dir.name[3:])

                # Check core type if available
                core_type_file = cpu_dir / "topology" / "core_type"
                if core_type_file.exists():
                    with open(core_type_file) as f:
                        core_type = int(f.read().strip(), 16)
                        if core_type == 0x40:  # P-core
                            p_cores.append(cpu_num)
                else:
                    # Assume first 8 cores are P-cores for Core Ultra
                    if cpu_num < self.num_p_cores:
                        p_cores.append(cpu_num)

            return p_cores[:self.num_p_cores] if p_cores else list(range(self.num_p_cores))

        else:
            # Windows or other: assume first N cores are P-cores
            return list(range(self.num_p_cores))

    def _detect_e_cores(self) -> List[int]:
        """Detect E-cores on the system"""
        try:
            import psutil
            total_cores = psutil.cpu_count(logical=False)
            # E-cores are everything after P-cores
            return list(range(self.num_p_cores, total_cores))
        except:
            # Assume 12 E-cores for Core Ultra 7 265K
            return list(range(self.num_p_cores, self.num_p_cores + 12))

    def assign_cores_to_trial(self, trial_id: int, num_trials: int) -> List[int]:
        """Assign P-cores to a trial based on current allocation strategy

        Args:
            trial_id: Trial identifier
            num_trials: Total number of trials

        Returns:
            List of core IDs assigned to this trial
        """
        if num_trials <= self.num_p_cores:
            # Each trial gets one or more P-cores
            cores_per_trial = self.num_p_cores // num_trials
            start_idx = trial_id * cores_per_trial
            end_idx = start_idx + cores_per_trial

            # Last trial gets any remaining cores
            if trial_id == num_trials - 1:
                end_idx = self.num_p_cores

            assigned_cores = self.p_core_ids[start_idx:end_idx]
        else:
            # More trials than cores: share cores
            assigned_cores = [self.p_core_ids[trial_id % self.num_p_cores]]

        self.trial_assignments[trial_id] = assigned_cores
        return assigned_cores

    def set_process_affinity(self, pid: Optional[int] = None, cores: List[int] = None):
        """Set process CPU affinity

        Args:
            pid: Process ID (None for current process)
            cores: List of core IDs to bind to
        """
        if cores is None:
            cores = self.p_core_ids

        try:
            import psutil

            process = psutil.Process(pid) if pid else psutil.Process()
            process.cpu_affinity(cores)

            # Set high priority
            if self.is_windows:
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                # Linux/Mac: negative nice values = higher priority
                try:
                    process.nice(-10)
                except:
                    pass  # May require root

            return True

        except Exception as e:
            print(f"Failed to set CPU affinity: {e}")
            return False

    def rotate_core_assignments(self) -> Dict[int, List[int]]:
        """Rotate core assignments to distribute thermal load

        Returns:
            New assignments after rotation
        """
        with self.rotation_lock:
            if not self.rotate_cores:
                return self.trial_assignments

            # Check if it's time to rotate
            now = datetime.now()
            if now - self.last_rotation < self.rotation_interval:
                return self.trial_assignments

            # Perform rotation
            self.rotation_count += 1
            rotation_offset = self.rotation_count * 2  # Rotate by 2 cores each time

            new_assignments = {}
            for trial_id, old_cores in self.trial_assignments.items():
                new_cores = []
                for core in old_cores:
                    # Find index in P-core list
                    idx = self.p_core_ids.index(core)
                    # Rotate index
                    new_idx = (idx + rotation_offset) % len(self.p_core_ids)
                    new_cores.append(self.p_core_ids[new_idx])

                new_assignments[trial_id] = new_cores

            self.trial_assignments = new_assignments
            self.last_rotation = now

            return new_assignments

    def optimize_for_windows(self):
        """Apply Windows-specific optimizations"""
        if not self.is_windows:
            return

        try:
            import subprocess

            # Set power plan to High Performance
            subprocess.run(
                ['powercfg', '/setactive', '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'],
                capture_output=True
            )

            # Disable CPU throttling
            subprocess.run([
                'powercfg', '/setacvalueindex', 'scheme_current',
                'sub_processor', 'procthrottlemax', '100'
            ], capture_output=True)

            # Disable core parking
            subprocess.run([
                'powercfg', '/setacvalueindex', 'scheme_current',
                'sub_processor', 'cpmincores', '100'
            ], capture_output=True)

            print("Windows power optimizations applied")

        except Exception as e:
            print(f"Failed to apply Windows optimizations: {e}")

    def optimize_for_linux(self):
        """Apply Linux-specific optimizations"""
        if not self.is_linux:
            return

        try:
            import subprocess

            # Set CPU governor to performance
            for cpu in self.p_core_ids:
                governor_file = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                try:
                    with open(governor_file, 'w') as f:
                        f.write('performance')
                except:
                    # Try with cpupower
                    subprocess.run(
                        ['cpupower', '-c', str(cpu), 'frequency-set', '-g', 'performance'],
                        capture_output=True
                    )

            print("Linux CPU governor set to performance")

        except Exception as e:
            print(f"Failed to apply Linux optimizations: {e}")

    def configure_numa_and_cache(self):
        """Configure NUMA and cache optimizations"""
        try:
            # Set environment variables for optimal NUMA/cache usage
            os.environ['OMP_NUM_THREADS'] = str(self.num_p_cores)
            os.environ['MKL_NUM_THREADS'] = str(self.num_p_cores)
            os.environ['NUMEXPR_NUM_THREADS'] = str(self.num_p_cores)

            # Intel-specific optimizations
            os.environ['KMP_AFFINITY'] = 'granularity=fine,explicit,proclist=[' + \
                                         ','.join(map(str, self.p_core_ids)) + ']'
            os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX512_E1'  # Arrow Lake supports this

            # Disable dynamic adjustment
            os.environ['MKL_DYNAMIC'] = 'FALSE'
            os.environ['OMP_DYNAMIC'] = 'FALSE'

        except Exception as e:
            print(f"Failed to configure NUMA/cache: {e}")

    def get_core_temperature(self, core_id: int) -> Optional[float]:
        """Get temperature of a specific core"""
        try:
            import psutil

            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                for sensor in temps['coretemp']:
                    if f'Core {core_id}' in sensor.label:
                        return sensor.current
        except:
            pass

        return None

    def get_thermal_status(self) -> Dict[str, any]:
        """Get thermal status of all P-cores"""
        status = {
            'temperatures': {},
            'avg_temp': 0,
            'max_temp': 0,
            'throttling': False
        }

        temps = []
        for core_id in self.p_core_ids:
            temp = self.get_core_temperature(core_id)
            if temp:
                status['temperatures'][core_id] = temp
                temps.append(temp)

        if temps:
            status['avg_temp'] = sum(temps) / len(temps)
            status['max_temp'] = max(temps)
            status['throttling'] = status['max_temp'] > 85  # Typical throttle temp

        return status

    def rebalance_after_elimination(self, remaining_trials: List[int]):
        """Rebalance core assignments after trial elimination

        Args:
            remaining_trials: List of trial IDs still active
        """
        num_remaining = len(remaining_trials)
        if num_remaining == 0:
            return

        # Redistribute all P-cores among remaining trials
        cores_per_trial = self.num_p_cores // num_remaining
        extra_cores = self.num_p_cores % num_remaining

        new_assignments = {}
        core_idx = 0

        for i, trial_id in enumerate(remaining_trials):
            num_cores = cores_per_trial
            if i < extra_cores:
                num_cores += 1

            assigned = self.p_core_ids[core_idx:core_idx + num_cores]
            new_assignments[trial_id] = assigned
            core_idx += num_cores

            # Update affinity if process exists
            self._update_trial_affinity(trial_id, assigned)

        self.trial_assignments = new_assignments

    def _update_trial_affinity(self, trial_id: int, cores: List[int]):
        """Update CPU affinity for a running trial"""
        # This would need integration with the trial process management
        # For now, just update the assignment
        pass

    def get_performance_config(self) -> Dict[str, str]:
        """Get environment variables for performance optimization"""
        return {
            'OMP_NUM_THREADS': str(self.num_p_cores),
            'MKL_NUM_THREADS': str(self.num_p_cores),
            'NUMEXPR_NUM_THREADS': str(self.num_p_cores),
            'KMP_AFFINITY': 'granularity=fine,explicit,proclist=[' +
                           ','.join(map(str, self.p_core_ids)) + ']',
            'MKL_ENABLE_INSTRUCTIONS': 'AVX512_E1',
            'MKL_DYNAMIC': 'FALSE',
            'OMP_DYNAMIC': 'FALSE'
        }

    def apply_optimizations(self):
        """Apply all platform-specific optimizations"""
        if self.is_windows:
            self.optimize_for_windows()
        elif self.is_linux:
            self.optimize_for_linux()

        self.configure_numa_and_cache()

        print(f"Core manager initialized:")
        print(f"  Platform: {self.platform}")
        print(f"  P-cores: {self.p_core_ids}")
        print(f"  E-cores: {self.e_core_ids[:4]}... ({len(self.e_core_ids)} total)")
        print(f"  Rotation: {'Enabled' if self.rotate_cores else 'Disabled'}")