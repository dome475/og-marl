#!/usr/bin/env python3
"""
Convert OG-MARL vault to JSON format
Use for small samples only (large datasets will be huge as JSON)
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path

def convert_vault_to_json(vault_path, output_dir, quality=None, max_timesteps=10000):
    """
    Convert vault to JSON format

    Args:
        vault_path: Path to .vlt directory
        output_dir: Output directory for .json files
        quality: Specific quality to convert (None = auto-detect first)
        max_timesteps: Maximum timesteps to export (to keep file size reasonable)
    """

    try:
        from flashbax.vault import Vault
        import jax
    except ImportError:
        print(f"Error: Missing dependencies. Install with:")
        print(f"  pip install flashbax jax numpy")
        sys.exit(1)

    vault_dir = Path(vault_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not vault_dir.exists():
        print(f"Error: Vault not found at {vault_path}")
        return

    scenario_name = vault_dir.stem.replace('.vlt', '')

    print(f"Vault: {vault_dir.name}")
    print(f"Output: {output_path}")
    print(f"Max timesteps: {max_timesteps}")
    print()

    # Find available qualities
    available_qualities = []
    for item in vault_dir.iterdir():
        if item.is_dir() and (item / 'metadata.json').exists():
            available_qualities.append(item.name)

    if not available_qualities:
        print("Error: No quality directories found in vault!")
        return

    # Select quality
    if quality:
        if quality not in available_qualities:
            print(f"Error: Quality '{quality}' not found!")
            print(f"Available: {available_qualities}")
            return
        quality_to_convert = quality
    else:
        quality_to_convert = available_qualities[0]
        print(f"Auto-selected quality: {quality_to_convert}")

    print(f"\nConverting quality: {quality_to_convert}")

    try:
        # Load vault
        print(f"  Loading vault data...")
        vault = Vault(str(vault_dir), vault_uid=quality_to_convert)
        experience = vault.read().experience
        data = jax.tree.map(lambda x: np.array(x), experience)

        # Extract metadata
        n_timesteps_total = data['observations'].shape[1]
        n_agents = data['observations'].shape[2]
        obs_dim = data['observations'].shape[-1]
        act_dim = data['actions'].shape[-1]

        print(f"  Total timesteps: {n_timesteps_total:,}")
        print(f"  Agents: {n_agents}")
        print(f"  Obs dim: {obs_dim}, Act dim: {act_dim}")

        # Limit timesteps
        n_timesteps = min(max_timesteps, n_timesteps_total)
        if n_timesteps < n_timesteps_total:
            print(f"  Exporting first {n_timesteps:,} timesteps (limited for file size)")
        else:
            print(f"  Exporting all {n_timesteps:,} timesteps")

        # Build trajectories
        print(f"  Building trajectory data...")
        trajectories = []

        for t in range(n_timesteps):
            step = {
                't': t,
                'obs': data['observations'][0, t].tolist(),
                'act': data['actions'][0, t].tolist(),
                'rew': data['rewards'][0, t].tolist()
            }

            # Add state if available
            if 'infos' in data and 'state' in data['infos']:
                step['state'] = data['infos']['state'][0, t].tolist()

            trajectories.append(step)

            # Progress indicator
            if (t + 1) % 1000 == 0:
                print(f"    Progress: {t+1:,}/{n_timesteps:,}")

        # Create output structure
        output_data = {
            'metadata': {
                'scenario': scenario_name,
                'quality': quality_to_convert,
                'n_agents': int(n_agents),
                'n_timesteps': int(n_timesteps),
                'n_timesteps_total': int(n_timesteps_total),
                'obs_dim': int(obs_dim),
                'act_dim': int(act_dim)
            },
            'trajectories': trajectories
        }

        # Save JSON
        output_file = output_path / f"{scenario_name}_{quality_to_convert}.json"
        print(f"  Saving to {output_file.name}...")

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {output_file.name} ({size_mb:.1f} MB)")

        if n_timesteps < n_timesteps_total:
            print(f"\n  Note: Only {n_timesteps}/{n_timesteps_total} timesteps exported")
            print(f"  Use --max-timesteps to export more (warning: large files!)")

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n✓ Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert OG-MARL vault to JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert first 10k timesteps (default)
  python vault_to_json.py data/2halfcheetah.vlt outputs/converted/

  # Convert specific quality, 5k timesteps
  python vault_to_json.py data/2halfcheetah.vlt outputs/converted/ --quality Replay --max-timesteps 5000

  # Convert first 1k timesteps for quick testing
  python vault_to_json.py data/2halfcheetah.vlt outputs/converted/ --max-timesteps 1000
        """
    )

    parser.add_argument('vault_path', help='Path to .vlt directory')
    parser.add_argument('output_dir', help='Output directory for .json files')
    parser.add_argument('--quality', help='Specific quality to convert')
    parser.add_argument('--max-timesteps', type=int, default=10000,
                       help='Maximum timesteps to export (default: 10000)')

    args = parser.parse_args()

    convert_vault_to_json(
        args.vault_path,
        args.output_dir,
        quality=args.quality,
        max_timesteps=args.max_timesteps
    )
