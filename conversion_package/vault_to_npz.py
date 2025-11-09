#!/usr/bin/env python3
"""
Convert OG-MARL vault to compressed NumPy format (.npz)
Efficient storage for full datasets
"""

import sys
import argparse
import numpy as np
from pathlib import Path

def convert_vault_to_npz(vault_path, output_dir, quality=None, all_qualities=False):
    """
    Convert vault to NPZ format

    Args:
        vault_path: Path to .vlt directory
        output_dir: Output directory for .npz files
        quality: Specific quality to convert (None = auto-detect first)
        all_qualities: Convert all available qualities
    """

    try:
        from flashbax.vault import Vault
        import jax
    except ImportError as e:
        print(f"Error: Missing dependencies. Install with:")
        print(f"  pip install flashbax jax numpy")
        sys.exit(1)

    # Check for Windows and warn
    import platform
    if platform.system() == 'Windows':
        print("⚠️  WARNING: Vault conversion does not work with Windows Python!")
        print("   The tensorstore library has issues with Windows paths.")
        print()
        print("   Please use WSL (Windows Subsystem for Linux) instead:")
        print(f"   1. Open WSL terminal")
        print(f"   2. Navigate to project: cd /mnt/c/Users/dbehl/Documents/GitHub/og-marl")
        print(f"   3. Run: python converters/vault_to_npz.py /mnt/d/OG_MARL/20_trains/20_trains.vlt /mnt/d/OG_MARL/20_trains/")
        print()
        print("   Continuing anyway (will likely fail)...")
        print()

    vault_dir = Path(vault_path).resolve()  # Get absolute path
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not vault_dir.exists():
        print(f"Error: Vault not found at {vault_path}")
        return

    scenario_name = vault_dir.stem.replace('.vlt', '')

    print(f"Vault: {vault_dir.name}")
    print(f"Output: {output_path}")
    print()

    # Find available qualities
    available_qualities = []
    for item in vault_dir.iterdir():
        if item.is_dir() and (item / 'metadata.json').exists():
            available_qualities.append(item.name)

    if not available_qualities:
        print("Error: No quality directories found in vault!")
        return

    print(f"Available qualities: {available_qualities}")

    # Determine which qualities to convert
    if all_qualities:
        qualities_to_convert = available_qualities
    elif quality:
        if quality not in available_qualities:
            print(f"Error: Quality '{quality}' not found!")
            print(f"Available: {available_qualities}")
            return
        qualities_to_convert = [quality]
    else:
        qualities_to_convert = [available_qualities[0]]
        print(f"Auto-selected quality: {qualities_to_convert[0]}")

    print()

    # Convert each quality
    for q in qualities_to_convert:
        print(f"Converting quality: {q}")

        try:
            # Load vault
            print(f"  Loading vault data...")
            vault = Vault(str(vault_dir), vault_uid=q)
            experience = vault.read().experience
            data = jax.tree.map(lambda x: np.array(x), experience)

            # Extract metadata
            n_timesteps = data['observations'].shape[1]
            n_agents = data['observations'].shape[2]
            obs_dim = data['observations'].shape[-1]
            act_dim = data['actions'].shape[-1]

            print(f"  Timesteps: {n_timesteps:,}")
            print(f"  Agents: {n_agents}")
            print(f"  Obs dim: {obs_dim}, Act dim: {act_dim}")

            # Prepare data dictionary (remove batch dimension)
            save_dict = {
                'observations': data['observations'][0],  # (timesteps, agents, obs_dim)
                'actions': data['actions'][0],            # (timesteps, agents, act_dim)
                'rewards': data['rewards'][0],            # (timesteps, agents)
                'n_timesteps': n_timesteps,
                'n_agents': n_agents,
                'obs_dim': obs_dim,
                'act_dim': act_dim,
                'scenario': scenario_name,
                'quality': q,
            }

            # Add state if available
            if 'infos' in data and 'state' in data['infos']:
                save_dict['states'] = data['infos']['state'][0]
                print(f"  State dim: {data['infos']['state'].shape[-1]}")

            # Save as compressed NPZ
            output_file = output_path / f"{scenario_name}_{q}.npz"
            print(f"  Saving to {output_file.name}...")

            np.savez_compressed(output_file, **save_dict)

            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved: {output_file.name} ({size_mb:.1f} MB)")
            print()

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue

    print(f"\n✓ Conversion complete!")
    print(f"\nConverted files saved to: {output_path}")
    print(f"\nTo load the data:")
    print(f"  import numpy as np")
    print(f"  data = np.load('path/to/file.npz')")
    print(f"  observations = data['observations']  # Shape: (timesteps, agents, obs_dim)")
    print(f"  actions = data['actions']            # Shape: (timesteps, agents, act_dim)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert OG-MARL vault to NumPy format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert default quality
  python vault_to_npz.py data/2halfcheetah.vlt outputs/converted/

  # Convert specific quality
  python vault_to_npz.py data/2halfcheetah.vlt outputs/converted/ --quality Replay

  # Convert all qualities
  python vault_to_npz.py data/2halfcheetah.vlt outputs/converted/ --all-qualities
        """
    )

    parser.add_argument('vault_path', help='Path to .vlt directory')
    parser.add_argument('output_dir', help='Output directory for .npz files')
    parser.add_argument('--quality', help='Specific quality to convert')
    parser.add_argument('--all-qualities', action='store_true',
                       help='Convert all available qualities')

    args = parser.parse_args()

    convert_vault_to_npz(
        args.vault_path,
        args.output_dir,
        quality=args.quality,
        all_qualities=args.all_qualities
    )
