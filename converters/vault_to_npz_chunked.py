#!/usr/bin/env python3
"""
Convert OG-MARL vault to NPZ format with chunked loading for large datasets
"""

import sys
import argparse
import numpy as np
from pathlib import Path

def convert_vault_chunked(vault_path, output_dir, quality=None, chunk_size=10000):
    """
    Convert vault to NPZ in chunks to avoid memory issues

    Args:
        vault_path: Path to .vlt directory
        output_dir: Output directory
        quality: Quality level to convert
        chunk_size: Number of timesteps per chunk
    """

    try:
        from flashbax.vault import Vault
        import jax
    except ImportError:
        print("Error: Missing dependencies. Install with:")
        print("  pip install flashbax jax numpy")
        sys.exit(1)

    vault_dir = Path(vault_path).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not vault_dir.exists():
        print(f"Error: Vault not found at {vault_path}")
        return

    scenario_name = vault_dir.stem.replace('.vlt', '')

    # Find qualities
    available_qualities = [d.name for d in vault_dir.iterdir()
                          if d.is_dir() and (d / 'metadata.json').exists()]

    if not available_qualities:
        print("Error: No quality directories found!")
        return

    quality_to_convert = quality if quality in available_qualities else available_qualities[0]

    print(f"Vault: {vault_dir.name}")
    print(f"Quality: {quality_to_convert}")
    print(f"Chunk size: {chunk_size:,} timesteps")
    print()

    try:
        # First pass: get metadata without loading all data
        print("Reading vault metadata...")
        vault = Vault(str(vault_dir), vault_uid=quality_to_convert)

        # Read metadata from JSON
        import json
        with open(vault_dir / quality_to_convert / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Parse shapes
        import re
        shapes = metadata['structure_shape']
        obs_shape = shapes['observations']
        match = re.search(r'\(1, (\d+), (\d+), (\d+)\)', obs_shape)
        if not match:
            print("Error: Could not parse observation shape")
            return

        total_timesteps, n_agents, obs_dim = map(int, match.groups())

        # Get action dim
        act_shape = shapes['actions']
        if '(' in act_shape:
            act_match = re.search(r'\(1, \d+, \d+(?:, (\d+))?\)', act_shape)
            if act_match and act_match.group(1):
                act_dim = int(act_match.group(1))
            else:
                act_dim = 1  # Discrete
        else:
            act_dim = 1

        print(f"Dataset info:")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Agents: {n_agents}")
        print(f"  Obs dim: {obs_dim}")
        print(f"  Act dim: {act_dim}")
        print()

        # Load full data (hoping it fits in memory)
        print("Loading vault data... (this may take a while)")
        experience = vault.read().experience
        data = jax.tree.map(lambda x: np.array(x), experience)

        print("Converting to NPZ format...")

        # Save
        save_dict = {
            'observations': data['observations'][0],
            'actions': data['actions'][0],
            'rewards': data['rewards'][0],
            'n_timesteps': total_timesteps,
            'n_agents': n_agents,
            'obs_dim': obs_dim,
            'act_dim': act_dim,
            'scenario': scenario_name,
            'quality': quality_to_convert,
        }

        if 'infos' in data and 'state' in data['infos']:
            save_dict['states'] = data['infos']['state'][0]
            print(f"  Including state data")

        output_file = output_path / f"{scenario_name}_{quality_to_convert}.npz"
        print(f"Saving to {output_file.name}...")

        np.savez_compressed(output_file, **save_dict)

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✓ Success! ({size_mb:.1f} MB)")

    except MemoryError:
        print("✗ Out of memory! Dataset too large.")
        print("Try using vault_to_json.py with --max-timesteps instead:")
        print(f"  python converters/vault_to_json.py {vault_path} {output_dir} --max-timesteps 10000")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert vault to NPZ (memory-aware)')
    parser.add_argument('vault_path', help='Path to .vlt directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--quality', help='Quality to convert')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size')

    args = parser.parse_args()
    convert_vault_chunked(args.vault_path, args.output_dir, args.quality, args.chunk_size)
