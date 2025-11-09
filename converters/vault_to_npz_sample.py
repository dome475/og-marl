#!/usr/bin/env python3
"""
Convert OG-MARL vault to NPZ by sampling subset of timesteps
Attempts to avoid loading full vault into memory
"""

import sys
import argparse
import numpy as np
from pathlib import Path

def convert_vault_sample(vault_path, output_dir, quality=None, max_timesteps=10000):
    """
    Sample subset of vault data

    Args:
        vault_path: Path to .vlt directory
        output_dir: Output directory
        quality: Quality level
        max_timesteps: Maximum timesteps to sample
    """

    try:
        from flashbax.vault import Vault
        import jax
    except ImportError:
        print("Error: Missing dependencies")
        sys.exit(1)

    vault_dir = Path(vault_path).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    scenario_name = vault_dir.stem.replace('.vlt', '')

    # Find quality
    available_qualities = [d.name for d in vault_dir.iterdir()
                          if d.is_dir() and (d / 'metadata.json').exists()]

    quality_to_convert = quality if quality in available_qualities else available_qualities[0]

    print(f"Vault: {vault_dir.name}")
    print(f"Quality: {quality_to_convert}")
    print(f"Max timesteps: {max_timesteps:,}")
    print()

    try:
        # Read metadata to get total size
        import json
        with open(vault_dir / quality_to_convert / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        print("Attempting to sample data...")
        vault = Vault(str(vault_dir), vault_uid=quality_to_convert)

        # Try to sample using flashbax's sample method
        try:
            # Sample max_timesteps using random indices
            total_timesteps = vault.read().experience['observations'].shape[1]
            print(f"Total timesteps in vault: {total_timesteps:,}")

            if max_timesteps >= total_timesteps:
                print("Requested timesteps >= total, loading all...")
                sampled_data = vault.read().experience
            else:
                print(f"Sampling {max_timesteps:,} timesteps from {total_timesteps:,}...")
                # Create sample indices (evenly spaced)
                indices = np.linspace(0, total_timesteps-1, max_timesteps, dtype=np.int32)

                # This will still load everything unfortunately
                full_data = vault.read().experience

                # Manually subsample
                sampled_data = {
                    'observations': full_data['observations'][:, indices],
                    'actions': full_data['actions'][:, indices],
                    'rewards': full_data['rewards'][:, indices],
                }

                if 'infos' in full_data:
                    sampled_data['infos'] = {}
                    if 'state' in full_data['infos']:
                        sampled_data['infos']['state'] = full_data['infos']['state'][:, indices]

        except Exception as e:
            print(f"Sampling failed: {e}")
            print("Trying full load...")
            sampled_data = vault.read().experience

        # Convert to numpy
        data = jax.tree.map(lambda x: np.array(x), sampled_data)

        # Get shapes
        n_timesteps = data['observations'].shape[1]
        n_agents = data['observations'].shape[2]
        obs_dim = data['observations'].shape[-1]

        print(f"Loaded: {n_timesteps:,} timesteps, {n_agents} agents")

        # Save
        save_dict = {
            'observations': data['observations'][0],
            'actions': data['actions'][0],
            'rewards': data['rewards'][0],
            'n_timesteps': n_timesteps,
            'n_agents': n_agents,
            'scenario': scenario_name,
            'quality': quality_to_convert,
        }

        if 'infos' in data and 'state' in data['infos']:
            save_dict['states'] = data['infos']['state'][0]

        output_file = output_path / f"{scenario_name}_{quality_to_convert}_sample{n_timesteps}.npz"
        print(f"Saving to {output_file.name}...")

        np.savez_compressed(output_file, **save_dict)

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✓ Success! ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample subset of vault')
    parser.add_argument('vault_path', help='Path to .vlt directory')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--quality', help='Quality level')
    parser.add_argument('--max-timesteps', type=int, default=10000,
                       help='Max timesteps to sample (default: 10000)')

    args = parser.parse_args()
    convert_vault_sample(args.vault_path, args.output_dir, args.quality, args.max_timesteps)
