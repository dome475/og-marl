#!/usr/bin/env python3
"""
Export joint action trajectories from Flashbax vaults to readable formats.
Uses proper tensorstore OCDBT format handling.
"""

import argparse
import json
import os
from pathlib import Path
import warnings

import numpy as np
import tensorstore as ts

warnings.filterwarnings('ignore')


def load_vault_metadata(vault_path: str):
    """Load vault metadata to understand structure."""
    metadata_path = Path(vault_path) / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def parse_shape_string(shape_str: str):
    """Parse shape string like '(1, 10000, 2, 3)' to tuple."""
    # Remove parentheses and split
    shape_str = shape_str.strip('()')
    return tuple(int(x.strip()) for x in shape_str.split(','))


def load_tensorstore_array(base_path: str, key_path: list):
    """Load a single array from tensorstore using OCDBT format."""
    # Construct the tensorstore path
    ts_path = str(Path(base_path)).replace('\\', '/')

    # Build the key path for nested structures
    full_key = '/'.join(key_path)

    # Tensorstore spec for OCDBT format
    spec = {
        'driver': 'ocdbt',
        'base': f'file://{ts_path}/',
        'path': full_key,
    }

    try:
        dataset = ts.open(spec).result()
        data = np.array(dataset.read().result())
        return data
    except Exception as e:
        print(f"Error loading {full_key}: {e}")
        # Try alternative spec
        try:
            spec = {
                'driver': 'ocdbt',
                'kvstore': {
                    'driver': 'file',
                    'path': ts_path
                },
                'path': full_key,
            }
            dataset = ts.open(spec).result()
            data = np.array(dataset.read().result())
            return data
        except Exception as e2:
            print(f"Alternative method also failed for {full_key}: {e2}")
            return None


def load_vault_data(vault_path: str):
    """Load all data from vault."""
    metadata = load_vault_metadata(vault_path)

    print(f"\nVault metadata:")
    print(f"  Version: {metadata['version']}")
    print(f"  Structure: {json.dumps(metadata['structure_shape'], indent=4)}")

    vault_data = {}

    # Load flat keys
    for key, shape_str in metadata['structure_shape'].items():
        if isinstance(shape_str, str):
            # Simple array
            print(f"Loading {key}...")
            data = load_tensorstore_array(vault_path, [key])
            if data is not None:
                vault_data[key] = data
                print(f"  ✓ Loaded {key}: shape {data.shape}")
        elif isinstance(shape_str, dict):
            # Nested structure (like 'infos')
            vault_data[key] = {}
            for subkey, subshape_str in shape_str.items():
                print(f"Loading {key}/{subkey}...")
                data = load_tensorstore_array(vault_path, [key, subkey])
                if data is not None:
                    vault_data[key][subkey] = data
                    print(f"  ✓ Loaded {key}/{subkey}: shape {data.shape}")

    return vault_data


def export_actions_to_json(vault_data: dict, output_path: str) -> None:
    """Export actions to JSON format."""
    actions = vault_data['actions']
    terminals = vault_data.get('terminals', None)

    # Remove batch dimension if present
    if len(actions.shape) == 4 and actions.shape[0] == 1:
        actions = actions[0]
    if terminals is not None and len(terminals.shape) == 3 and terminals.shape[0] == 1:
        terminals = terminals[0]

    print(f"\nProcessing actions with shape: {actions.shape}")

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
        'num_agents': actions.shape[1],
        'action_dim': actions.shape[2] if len(actions.shape) > 2 else 1,
        'num_episodes': len(episodes),
        'episodes': episodes
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Exported {len(episodes)} episodes to {output_path}")


def export_actions_to_csv(vault_data: dict, output_path: str) -> None:
    """Export actions to CSV format."""
    import csv

    actions = vault_data['actions']
    terminals = vault_data.get('terminals', None)

    # Remove batch dimension if present
    if len(actions.shape) == 4 and actions.shape[0] == 1:
        actions = actions[0]
    if terminals is not None and len(terminals.shape) == 3 and terminals.shape[0] == 1:
        terminals = terminals[0]

    num_timesteps = len(actions)
    num_agents = actions.shape[1]
    action_dim = actions.shape[2] if len(actions.shape) > 2 else 1

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        header = ['timestep', 'episode']
        for agent_idx in range(num_agents):
            if action_dim > 1:
                for action_idx in range(action_dim):
                    header.append(f'agent_{agent_idx}_action_{action_idx}')
            else:
                header.append(f'agent_{agent_idx}_action')
        if terminals is not None:
            header.append('episode_end')
        writer.writerow(header)

        # Write data
        episode_num = 0
        for t in range(num_timesteps):
            row = [t, episode_num]

            # Flatten actions for all agents
            for agent_idx in range(num_agents):
                if action_dim > 1:
                    row.extend(actions[t, agent_idx].tolist())
                else:
                    row.append(float(actions[t, agent_idx]))

            if terminals is not None:
                is_terminal = bool(terminals[t].any())
                row.append(is_terminal)
                if is_terminal:
                    episode_num += 1
            writer.writerow(row)

    print(f"✓ Exported {num_timesteps} timesteps to {output_path}")


def export_actions_to_txt(vault_data: dict, output_path: str, max_episodes: int = 10) -> None:
    """Export actions to human-readable text format."""
    actions = vault_data['actions']
    terminals = vault_data.get('terminals', None)
    rewards = vault_data.get('rewards', None)

    # Remove batch dimension if present
    if len(actions.shape) == 4 and actions.shape[0] == 1:
        actions = actions[0]
    if terminals is not None and len(terminals.shape) == 3 and terminals.shape[0] == 1:
        terminals = terminals[0]
    if rewards is not None and len(rewards.shape) == 3 and rewards.shape[0] == 1:
        rewards = rewards[0]

    num_agents = actions.shape[1]
    action_dim = actions.shape[2] if len(actions.shape) > 2 else 1

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("JOINT ACTION TRAJECTORIES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Number of agents: {num_agents}\n")
        f.write(f"Action dimension: {action_dim}\n")
        f.write(f"Total timesteps: {len(actions)}\n")
        f.write("=" * 80 + "\n\n")

        episode_num = 0
        episode_start = 0
        episode_return = np.zeros(num_agents)
        episodes_written = 0

        for t in range(len(actions)):
            if t == episode_start:
                if episodes_written >= max_episodes:
                    f.write(f"\n... (showing first {max_episodes} episodes only) ...\n")
                    break

                f.write(f"\n{'=' * 80}\n")
                f.write(f"EPISODE {episode_num}\n")
                f.write(f"{'=' * 80}\n")

            if episodes_written < max_episodes:
                f.write(f"\nTimestep {t} (Episode step {t - episode_start}):\n")
                for agent_idx in range(num_agents):
                    f.write(f"  Agent {agent_idx} action: {actions[t, agent_idx].tolist()}\n")

                if rewards is not None:
                    f.write(f"  Rewards: {rewards[t].tolist()}\n")
                    episode_return += rewards[t]

            if terminals is not None and terminals[t].any():
                if episodes_written < max_episodes:
                    f.write(f"\n  >>> EPISODE END <<<\n")
                    f.write(f"  Episode Length: {t - episode_start + 1}\n")
                    if rewards is not None:
                        f.write(f"  Episode Return: {episode_return.tolist()}\n")

                episode_num += 1
                episodes_written += 1
                episode_start = t + 1
                episode_return = np.zeros(num_agents)

    print(f"✓ Exported readable text format to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Export joint action trajectories from Flashbax vaults to readable formats'
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

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load vault
    print(f"Loading vault from {args.vault_path}...")

    try:
        vault_data = load_vault_data(args.vault_path)
    except Exception as e:
        print(f"Error loading vault: {e}")
        import traceback
        traceback.print_exc()
        return

    if not vault_data or 'actions' not in vault_data:
        print("Error: Could not load action data from vault")
        return

    # Print summary
    actions = vault_data['actions']
    if len(actions.shape) == 4:
        actions = actions[0]

    print(f"\n{'=' * 80}")
    print(f"VAULT SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total timesteps: {actions.shape[0]}")
    print(f"  Number of agents: {actions.shape[1]}")
    print(f"  Action dimension per agent: {actions.shape[2] if len(actions.shape) > 2 else 1}")
    print(f"  Full action shape: {actions.shape}")
    print()

    # Export to requested format(s)
    vault_name = Path(args.vault_path).parent.stem
    dataset_name = Path(args.vault_path).stem
    base_name = f"{vault_name}_{dataset_name}"

    if args.format in ['csv', 'all']:
        csv_path = os.path.join(args.output_dir, f'{base_name}_actions.csv')
        export_actions_to_csv(vault_data, csv_path)

    if args.format in ['json', 'all']:
        json_path = os.path.join(args.output_dir, f'{base_name}_actions.json')
        export_actions_to_json(vault_data, json_path)

    if args.format in ['txt', 'all']:
        txt_path = os.path.join(args.output_dir, f'{base_name}_actions.txt')
        export_actions_to_txt(vault_data, txt_path)

    print(f"\n{'=' * 80}")
    print(f"✓ All exports completed!")
    print(f"  Output directory: {os.path.abspath(args.output_dir)}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
