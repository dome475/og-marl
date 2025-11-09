#!/usr/bin/env python3
"""
Export joint action trajectories from vaults to readable formats.
Windows-compatible version with direct tensorstore reading.
"""

import argparse
import json
import os
from pathlib import Path
import warnings

import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')


def load_vault_data_direct(vault_path: str):
    """Load vault data using direct tensorstore access."""
    import tensorstore as ts

    # Find all data files
    replay_dir = Path(vault_path)

    # Look for the structure
    data_files = {}

    # Navigate the vault structure
    for item in replay_dir.rglob('*'):
        if item.is_file() and not item.name.startswith('.'):
            # Get relative path from vault root
            rel_path = item.relative_to(replay_dir)
            parts = list(rel_path.parts)
            if len(parts) >= 2:
                key = parts[0]  # e.g., 'actions', 'observations'
                data_files[key] = item.parent

    # Load each data array
    vault_data = {}

    for key, data_dir in data_files.items():
        try:
            # Use POSIX path for tensorstore
            spec = {
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'file',
                    'path': str(data_dir).replace('\\', '/')
                }
            }
            dataset = ts.open(spec).result()
            vault_data[key] = np.array(dataset.read().result())
            print(f"Loaded {key}: shape {vault_data[key].shape}")
        except Exception as e:
            print(f"Warning: Could not load {key}: {e}")

    return vault_data


def load_vault_data_simple(vault_path: str):
    """Try loading vault with simpler approach."""
    from flashbax.vault import Vault

    # Convert to POSIX path
    vault_path_posix = str(Path(vault_path)).replace('\\', '/')

    # Try loading with absolute path
    import os
    os.chdir(Path(vault_path).parent.parent.parent)

    vault_name = str(Path(vault_path).relative_to(Path.cwd())).replace('\\', '/')
    vault_uid = Path(vault_path).name

    vault = Vault(
        vault_name=vault_name.replace(f'/{vault_uid}', ''),
        vault_uid=vault_uid,
        rel_dir='.'
    )

    all_data = vault.read()
    return all_data.experience


def export_actions_to_json(vault_data: dict, output_path: str) -> None:
    """Export actions to JSON format."""
    # Handle different possible structures
    if isinstance(vault_data, dict):
        if 'actions' in vault_data:
            actions = np.array(vault_data['actions'])
            terminals = np.array(vault_data.get('terminals', None))
        else:
            print("Available keys:", list(vault_data.keys()))
            return
    else:
        actions = np.array(vault_data)
        terminals = None

    # Remove batch dimension if present
    if len(actions.shape) == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if terminals is not None and len(terminals.shape) == 3 and terminals.shape[0] == 1:
        terminals = terminals[0]

    print(f"Processing actions with shape: {actions.shape}")

    # Split into episodes
    episodes = []
    episode_actions = []

    for t in range(len(actions)):
        episode_actions.append(actions[t].tolist())

        if terminals is not None and terminals[t].any():
            episodes.append({
                'episode_num': len(episodes),
                'length': len(episode_actions),
                'actions': episode_actions
            })
            episode_actions = []

    # Add remaining actions if episode didn't terminate
    if episode_actions:
        episodes.append({
            'episode_num': len(episodes),
            'length': len(episode_actions),
            'actions': episode_actions,
            'incomplete': True
        })

    output_data = {
        'total_timesteps': len(actions),
        'num_agents': actions.shape[1] if len(actions.shape) > 1 else 1,
        'num_episodes': len(episodes),
        'episodes': episodes
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Exported {len(episodes)} episodes to {output_path}")


def export_actions_to_csv(vault_data: dict, output_path: str) -> None:
    """Export actions to CSV format."""
    import csv

    actions = np.array(vault_data['actions'])
    terminals = np.array(vault_data.get('terminals', None))

    # Remove batch dimension if present
    if len(actions.shape) == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if terminals is not None and len(terminals.shape) == 3 and terminals.shape[0] == 1:
        terminals = terminals[0]

    num_timesteps = len(actions)
    num_agents = actions.shape[1] if len(actions.shape) > 1 else 1

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = ['timestep', 'episode']
        header.extend([f'agent_{i}_action' for i in range(num_agents)])
        if terminals is not None:
            header.append('episode_end')
        writer.writerow(header)

        # Write data
        episode_num = 0
        for t in range(num_timesteps):
            row = [t, episode_num]
            if len(actions.shape) > 1:
                row.extend(actions[t].tolist())
            else:
                row.append(float(actions[t]))

            if terminals is not None:
                is_terminal = bool(terminals[t].any())
                row.append(is_terminal)
                if is_terminal:
                    episode_num += 1
            writer.writerow(row)

    print(f"✓ Exported {num_timesteps} timesteps to {output_path}")


def export_actions_to_txt(vault_data: dict, output_path: str) -> None:
    """Export actions to human-readable text format."""
    actions = np.array(vault_data['actions'])
    terminals = np.array(vault_data.get('terminals', None))
    rewards = np.array(vault_data.get('rewards', None))

    # Remove batch dimension if present
    if len(actions.shape) == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if terminals is not None and len(terminals.shape) == 3 and terminals.shape[0] == 1:
        terminals = terminals[0]
    if rewards is not None and len(rewards.shape) == 3 and rewards.shape[0] == 1:
        rewards = rewards[0]

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("JOINT ACTION TRAJECTORIES\n")
        f.write("=" * 80 + "\n\n")

        episode_num = 0
        episode_start = 0
        episode_return = np.zeros(actions.shape[1] if len(actions.shape) > 1 else 1)

        for t in range(len(actions)):
            if t == episode_start:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"EPISODE {episode_num}\n")
                f.write(f"{'=' * 80}\n")

            f.write(f"\nTimestep {t} (Episode step {t - episode_start}):\n")
            f.write(f"  Joint Action: {actions[t].tolist()}\n")

            if rewards is not None:
                f.write(f"  Rewards: {rewards[t].tolist()}\n")
                episode_return += rewards[t]

            if terminals is not None and terminals[t].any():
                f.write(f"\n  >>> EPISODE END <<<\n")
                f.write(f"  Episode Length: {t - episode_start + 1}\n")
                if rewards is not None:
                    f.write(f"  Episode Return: {episode_return.tolist()}\n")
                episode_num += 1
                episode_start = t + 1
                episode_return = np.zeros(actions.shape[1] if len(actions.shape) > 1 else 1)

    print(f"✓ Exported readable text format to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export joint action trajectories from vaults to readable formats'
    )
    parser.add_argument(
        '--vault-path',
        type=str,
        default='vaults/og_marl/gymnasium_mamujoco/2halfcheetah.vlt/Replay',
        help='Path to the vault directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='exported_actions',
        help='Output directory for exported files'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'json', 'txt', 'all'],
        default='json',
        help='Export format'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['direct', 'flashbax'],
        default='direct',
        help='Loading method (direct uses tensorstore, flashbax uses Vault API)'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load vault
    print(f"Loading vault from {args.vault_path}...")

    try:
        if args.method == 'direct':
            vault_data = load_vault_data_direct(args.vault_path)
        else:
            vault_data = load_vault_data_simple(args.vault_path)
            # Convert to dict if needed
            if not isinstance(vault_data, dict):
                vault_data = {
                    'actions': vault_data['actions'],
                    'terminals': vault_data.get('terminals'),
                    'rewards': vault_data.get('rewards')
                }
    except Exception as e:
        print(f"Error loading vault: {e}")
        print("\nTrying alternative method...")
        try:
            vault_data = load_vault_data_direct(args.vault_path)
        except Exception as e2:
            print(f"Failed with alternative method too: {e2}")
            return

    if not vault_data or 'actions' not in vault_data:
        print("Error: Could not load action data from vault")
        return

    # Print summary
    actions = np.array(vault_data['actions'])
    if len(actions.shape) == 3:
        actions = actions[0]
    print(f"\nVault Summary:")
    print(f"  Total timesteps: {actions.shape[0]}")
    print(f"  Number of agents: {actions.shape[1] if len(actions.shape) > 1 else 1}")
    print(f"  Action shape: {actions.shape}")
    print()

    # Export to requested format(s)
    base_name = Path(args.vault_path).parent.stem + "_" + Path(args.vault_path).stem

    if args.format in ['csv', 'all']:
        csv_path = os.path.join(args.output_dir, f'{base_name}_actions.csv')
        export_actions_to_csv(vault_data, csv_path)

    if args.format in ['json', 'all']:
        json_path = os.path.join(args.output_dir, f'{base_name}_actions.json')
        export_actions_to_json(vault_data, json_path)

    if args.format in ['txt', 'all']:
        txt_path = os.path.join(args.output_dir, f'{base_name}_actions.txt')
        export_actions_to_txt(vault_data, txt_path)

    print(f"\n✓ All exports completed! Files saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
