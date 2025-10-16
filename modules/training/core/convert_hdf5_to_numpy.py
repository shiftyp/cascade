#!/usr/bin/env python3
"""
Convert HDF5 cache to numpy memmap format for thread-safe parallel loading.

Numpy memmap allows multiple DataLoader workers to read simultaneously without serialization.
This dramatically improves training speed with multi-worker DataLoaders.
"""

import h5py
import numpy as np
from pathlib import Path
import json


def convert_hdf5_to_numpy(hdf5_path, output_dir=None):
    """
    Convert HDF5 cache to numpy memmap files.

    Creates:
    - streams.npy (memory-mapped)
    - embeddings.npy (memory-mapped)
    - metadata.json (for dataset info)

    Args:
        hdf5_path: Path to HDF5 cache file
        output_dir: Output directory (default: same as HDF5 with .npy suffix)
    """
    hdf5_path = Path(hdf5_path)

    if output_dir is None:
        output_dir = hdf5_path.parent / f"{hdf5_path.stem}_numpy"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting HDF5 → Numpy memmap:")
    print(f"  Input: {hdf5_path}")
    print(f"  Output: {output_dir}")

    with h5py.File(hdf5_path, 'r') as f:
        # Check if multi-file
        is_multifile = 'num_chunks' in f.attrs

        if is_multifile:
            print(f"\n  Multi-file dataset detected")
            num_chunks = f.attrs['num_chunks']
            chunk_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                          for name in f['chunk_files'][:]]

            # Load all chunks
            all_streams = []
            all_embeddings = []

            for i, chunk_name in enumerate(chunk_names):
                print(f"  Loading chunk {i+1}/{len(chunk_names)}: {chunk_name}")
                chunk_path = hdf5_path.parent / chunk_name

                with h5py.File(chunk_path, 'r') as cf:
                    all_streams.append(cf['streams'][:])
                    if 'optimal_embeddings' in cf:
                        all_embeddings.append(cf['optimal_embeddings'][:])

            streams = np.concatenate(all_streams, axis=0)
            if all_embeddings:
                embeddings = np.concatenate(all_embeddings, axis=0)
            else:
                embeddings = None

            metadata = dict(f.attrs)
        else:
            # Single file
            print(f"\n  Single-file dataset")
            streams = f['streams'][:]
            embeddings = f['optimal_embeddings'][:] if 'optimal_embeddings' in f else None

            metadata = {
                'num_streams': len(streams),
                'stream_duration_sec': f.attrs.get('stream_duration_sec', 10.0),
                'window_duration_sec': f.attrs.get('window_duration_sec', 2.0),
                'sample_rate': f.attrs.get('sample_rate', 48000),
                'windows_per_stream': f.attrs.get('windows_per_stream', 5),
            }

    # Save as numpy memmap files
    print(f"\n  Writing numpy files...")

    # Streams (memory-mapped for efficient loading)
    streams_path = output_dir / 'streams.npy'
    np.save(streams_path, streams)
    print(f"    ✓ Streams: {streams_path} ({streams.nbytes / 1e9:.1f} GB)")

    # Embeddings
    if embeddings is not None:
        embeddings_path = output_dir / 'embeddings.npy'
        np.save(embeddings_path, embeddings)
        print(f"    ✓ Embeddings: {embeddings_path} ({embeddings.nbytes / 1e6:.1f} MB)")

    # Metadata (JSON for easy reading)
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        # Convert numpy types to Python types for JSON
        metadata_clean = {}
        for k, v in metadata.items():
            if isinstance(v, (np.integer, np.floating)):
                metadata_clean[k] = v.item()
            elif isinstance(v, np.ndarray):
                metadata_clean[k] = v.tolist()
            else:
                metadata_clean[k] = v

        json.dump(metadata_clean, f, indent=2)
    print(f"    ✓ Metadata: {metadata_path}")

    print(f"\n✅ Conversion complete!")
    print(f"\nTo use numpy dataset:")
    print(f"  1. Set: export CASCADE_USE_NUMPY_CACHE=true")
    print(f"  2. Run training (DataLoader workers will read in parallel)")
    print(f"\nBenefits:")
    print(f"  ✓ Thread-safe parallel reading (no serialization!)")
    print(f"  ✓ Fast memory-mapped access")
    print(f"  ✓ Full GPU utilization with DataLoader workers")

    return output_dir


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_hdf5_to_numpy.py <hdf5_file>")
        print("\nExample:")
        print("  python convert_hdf5_to_numpy.py dataset_cache/streaming_cascade_v9_final_n200000streams_10.0s_seed42.h5")
        sys.exit(1)

    hdf5_file = sys.argv[1]
    convert_hdf5_to_numpy(hdf5_file)
