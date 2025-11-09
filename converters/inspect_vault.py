#!/usr/bin/env python3
"""
Inspect OG-MARL vault to see available qualities and metadata
"""

import sys
import json
from pathlib import Path

def inspect_vault(vault_path):
    """
    Inspect vault and display available qualities and metadata

    Args:
        vault_path: Path to .vlt directory
    """

    vault_dir = Path(vault_path)

    if not vault_dir.exists():
        print(f"Error: Vault not found at {vault_path}")
        return

    if not vault_dir.name.endswith('.vlt'):
        print(f"Warning: Path doesn't end with .vlt: {vault_path}")

    print(f"Vault: {vault_dir.name}")
    print(f"Path: {vault_dir}")
    print()

    # Find all quality directories
    qualities = []
    for item in vault_dir.iterdir():
        if item.is_dir():
            # Check if it has the vault structure (metadata.json, manifest.ocdbt, d/)
            has_metadata = (item / 'metadata.json').exists()
            has_manifest = (item / 'manifest.ocdbt').exists()
            has_data_dir = (item / 'd').exists()

            if has_metadata or has_manifest or has_data_dir:
                qualities.append({
                    'name': item.name,
                    'path': item,
                    'has_metadata': has_metadata,
                    'has_manifest': has_manifest,
                    'has_data': has_data_dir
                })

    if not qualities:
        print("No quality directories found!")
        print(f"\nContents of {vault_dir}:")
        for item in vault_dir.iterdir():
            print(f"  {item.name}")
        return

    print(f"Found {len(qualities)} quality level(s):")
    print()

    for i, q in enumerate(qualities, 1):
        print(f"{i}. Quality: {q['name']}")
        print(f"   Path: {q['path']}")
        print(f"   Structure:")
        print(f"     - metadata.json: {'✓' if q['has_metadata'] else '✗'}")
        print(f"     - manifest.ocdbt: {'✓' if q['has_manifest'] else '✗'}")
        print(f"     - d/ (data): {'✓' if q['has_data'] else '✗'}")

        # Try to read metadata
        if q['has_metadata']:
            try:
                with open(q['path'] / 'metadata.json', 'r') as f:
                    metadata = json.load(f)
                print(f"   Metadata:")
                print(f"     - version: {metadata.get('version', 'unknown')}")

                # Parse structure shape for more details
                if 'structure_shape' in metadata:
                    shapes = metadata['structure_shape']
                    print(f"     - Data shapes:")

                    # Extract key info from observations
                    if 'observations' in shapes:
                        obs_shape = shapes['observations']
                        # Parse shape string like "(1, 50000, 20, 238)"
                        import re
                        match = re.search(r'\(1, (\d+), (\d+), (\d+)\)', obs_shape)
                        if match:
                            timesteps, agents, obs_dim = match.groups()
                            print(f"         Timesteps: {int(timesteps):,}")
                            print(f"         Agents: {agents}")
                            print(f"         Obs dim: {obs_dim}")

                    # Actions
                    if 'actions' in shapes:
                        act_shape = shapes['actions']
                        # Try continuous actions: (1, timesteps, agents, act_dim)
                        match_cont = re.search(r'\(1, \d+, \d+, (\d+)\)', act_shape)
                        # Try discrete actions: (1, timesteps, agents)
                        match_disc = re.search(r'\(1, (\d+), (\d+)\)$', act_shape)

                        if match_cont:
                            print(f"         Act dim: {match_cont.group(1)} (continuous)")
                        elif match_disc:
                            # Discrete actions - check dtype to confirm
                            dtypes = metadata.get('structure_dtype', {})
                            act_dtype = dtypes.get('actions', 'unknown')
                            print(f"         Actions: discrete ({act_dtype})")
                        else:
                            print(f"         Actions: {act_shape}")

                    # State
                    if 'infos' in shapes and isinstance(shapes['infos'], dict) and 'state' in shapes['infos']:
                        state_shape = shapes['infos']['state']
                        match = re.search(r'\(1, \d+, (\d+)\)', state_shape)
                        if match:
                            print(f"         State dim: {match.group(1)}")

                # Show data types
                if 'structure_dtype' in metadata:
                    dtypes = metadata['structure_dtype']
                    if 'observations' in dtypes:
                        print(f"     - Obs dtype: {dtypes['observations']}")
                    if 'actions' in dtypes:
                        print(f"     - Act dtype: {dtypes['actions']}")

            except Exception as e:
                print(f"   Metadata: Error reading ({e})")

        # Check data directory size
        if q['has_data']:
            data_dir = q['path'] / 'd'
            num_files = len(list(data_dir.iterdir()))
            print(f"   Data files: {num_files}")

        print()

    print("To convert this vault, use:")
    print(f"  python converters/vault_to_npz.py {vault_dir} outputs/converted/")
    print(f"  python converters/vault_to_npz.py {vault_dir} outputs/converted/ --all-qualities")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_vault.py <vault_path>")
        print()
        print("Example:")
        print("  python inspect_vault.py data/2halfcheetah.vlt")
        print("  python inspect_vault.py /mnt/c/Users/user/Downloads/2halfcheetah.vlt")
        sys.exit(1)

    vault_path = sys.argv[1]
    inspect_vault(vault_path)
