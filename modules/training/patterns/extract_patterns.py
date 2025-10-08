"""Extract individual pattern files from tournament checkpoint (3-FSK ternary patterns)"""

import pickle
import numpy as np
from pathlib import Path
import sys

# Find latest checkpoint file
checkpoint_dir = Path('/workspaces/cascade/modules/training/patterns/tournament/checkpoints')

# Allow command line argument or auto-detect
if len(sys.argv) > 1:
    master_file = Path(sys.argv[1])
else:
    # Find most recent checkpoint output
    output_dirs = list(checkpoint_dir.glob('*/output'))
    if not output_dirs:
        print("ERROR: No checkpoint output directories found")
        sys.exit(1)

    latest_dir = max(output_dirs, key=lambda p: p.stat().st_mtime)
    pattern_files = list(latest_dir.glob('patterns_*.pkl'))

    if not pattern_files:
        print(f"ERROR: No pattern files found in {latest_dir}")
        sys.exit(1)

    master_file = pattern_files[0]

output_dir = Path('/workspaces/cascade/modules/training/patterns/tournament')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading master pattern file: {master_file}")
with open(master_file, 'rb') as f:
    data = pickle.load(f)

print(f"Master file keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")

# Extract nested patterns
if 'nested_patterns' in data:
    nested = data['nested_patterns']
    print(f"Nested pattern lengths: {list(nested.keys())}")

    # For each length, extract all 8 pattern IDs
    for length in [64, 128, 256, 512, 1024, 2048]:
        if length in nested:
            cores = nested[length]['cores']
            print(f"\nLength {length}: {len(cores)} patterns")

            for pattern_id in range(len(cores)):
                pattern_symbols = cores[pattern_id]

                # Verify it's the right format
                pattern_array = np.asarray(pattern_symbols, dtype=np.uint8)
                assert pattern_array.shape == (length,), f"Wrong shape: {pattern_array.shape}"

                # Check if ternary (3-FSK) or binary (2-FSK)
                unique_vals = np.unique(pattern_array)
                is_ternary = len(unique_vals) <= 3 and np.all((pattern_array >= 0) & (pattern_array <= 2))
                is_binary = len(unique_vals) <= 2 and np.all((pattern_array >= 0) & (pattern_array <= 1))

                if is_ternary:
                    modulation = "3-FSK (ternary)"
                elif is_binary:
                    modulation = "2-FSK (binary)"
                else:
                    print(f"  WARNING: Pattern {pattern_id} has unexpected values: {unique_vals}")
                    modulation = "unknown"

                # Save individual pattern file
                output_file = output_dir / f"pattern_{pattern_id}_{length}.pkl"
                with open(output_file, 'wb') as f_out:
                    pickle.dump(pattern_array, f_out)

                # Verify statistics
                symbol_counts = {val: np.sum(pattern_array == val) for val in unique_vals}
                print(f"  Pattern {pattern_id} ({modulation}): {symbol_counts}")
        else:
            print(f"WARNING: Length {length} not found in master file")
else:
    print("ERROR: 'nested_patterns' key not found in master file")
    print(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")

print(f"\n✓ Extracted patterns to: {output_dir}")
print(f"Total files created: {len(list(output_dir.glob('pattern_*.pkl')))}")
