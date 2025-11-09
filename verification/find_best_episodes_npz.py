#!/usr/bin/env python3
"""
Analyze OG-MARL NPZ data to find best performing episodes
"""

import json
import numpy as np
from pathlib import Path
import sys

def find_best_episodes(npz_file, episode_length=1000, top_n=5):
    """
    Analyze NPZ dataset and find episodes with highest cumulative rewards

    Args:
        npz_file: Path to exported NPZ file
        episode_length: Length of episode window to analyze
        top_n: Number of top episodes to identify
    """

    print(f"📂 Loading {npz_file}")
    data = np.load(npz_file)

    # Extract metadata
    env = str(data['env'])
    scenario = str(data['scenario'])
    quality = str(data['quality'])
    n_timesteps = int(data['n_timesteps'])
    n_agents = int(data['n_agents'])

    print(f"\n📊 Dataset Info:")
    print(f"   Environment: {env}")
    print(f"   Scenario: {scenario}")
    print(f"   Quality: {quality}")
    print(f"   Total timesteps: {n_timesteps}")
    print(f"   Agents: {n_agents}")

    # Load trajectory data
    observations = data['observations']  # Shape: (timesteps, agents, obs_dim)
    actions = data['actions']            # Shape: (timesteps, agents, act_dim)
    rewards = data['rewards']            # Shape: (timesteps, agents)

    print(f"   Shapes: obs={observations.shape}, act={actions.shape}, rew={rewards.shape}")

    print(f"\n🔍 Analyzing trajectories in {episode_length}-step windows...")

    # Calculate cumulative rewards for sliding windows
    episode_rewards = []
    episode_starts = []

    step_size = episode_length // 2  # 50% overlap

    for start in range(0, n_timesteps - episode_length, step_size):
        episode_reward = np.sum(rewards[start:start + episode_length])
        episode_rewards.append(episode_reward)
        episode_starts.append(start)

    episode_rewards = np.array(episode_rewards)

    # Find statistics
    print(f"\n📈 Reward Statistics (per {episode_length} steps):")
    print(f"   Total episodes analyzed: {len(episode_rewards)}")
    print(f"   Mean cumulative reward: {np.mean(episode_rewards):.2f}")
    print(f"   Std dev: {np.std(episode_rewards):.2f}")
    print(f"   Min reward: {np.min(episode_rewards):.2f}")
    print(f"   Max reward: {np.max(episode_rewards):.2f}")

    # Find best episodes
    best_indices = np.argsort(episode_rewards)[-top_n:][::-1]

    print(f"\n🏆 Top {top_n} Best Episodes:")
    for rank, idx in enumerate(best_indices, 1):
        start_step = episode_starts[idx]
        reward = episode_rewards[idx]
        avg_reward = reward / episode_length
        print(f"   #{rank}: Steps {start_step:7d}-{start_step + episode_length:7d} | Total: {reward:8.2f} | Avg: {avg_reward:6.3f}")

    # Find worst episodes
    worst_indices = np.argsort(episode_rewards)[:top_n]

    print(f"\n😞 Worst {top_n} Episodes (for comparison):")
    for rank, idx in enumerate(worst_indices, 1):
        start_step = episode_starts[idx]
        reward = episode_rewards[idx]
        avg_reward = reward / episode_length
        print(f"   #{rank}: Steps {start_step:7d}-{start_step + episode_length:7d} | Total: {reward:8.2f} | Avg: {avg_reward:6.3f}")

    # Export best episode as JSON for visualization
    best_episode_start = episode_starts[best_indices[0]]
    best_episode_end = best_episode_start + episode_length

    print(f"\n💾 Exporting best episode to JSON for visualization...")

    trajectories = []
    for t in range(best_episode_start, best_episode_end):
        step = {
            't': int(t - best_episode_start),
            'obs': observations[t].tolist(),
            'act': actions[t].tolist(),
            'rew': rewards[t].tolist()
        }
        if 'states' in data:
            step['state'] = data['states'][t].tolist()
        trajectories.append(step)

    best_episode_data = {
        'metadata': {
            'env': env,
            'scenario': scenario,
            'quality': quality,
            'n_agents': n_agents,
            'n_timesteps': episode_length,
            'obs_dim': observations.shape[-1],
            'act_dim': actions.shape[-1],
            'note': f'Best episode: steps {best_episode_start}-{best_episode_end}',
            'original_start_step': best_episode_start,
            'cumulative_reward': float(episode_rewards[best_indices[0]])
        },
        'trajectories': trajectories
    }

    output_file = Path(npz_file).parent / f"{Path(npz_file).stem}_BEST.json"
    with open(output_file, 'w') as f:
        json.dump(best_episode_data, f, indent=2)

    print(f"   ✅ Saved: {output_file.name}")

    # Also export worst for comparison
    worst_episode_start = episode_starts[worst_indices[0]]
    worst_episode_end = worst_episode_start + episode_length

    trajectories = []
    for t in range(worst_episode_start, worst_episode_end):
        step = {
            't': int(t - worst_episode_start),
            'obs': observations[t].tolist(),
            'act': actions[t].tolist(),
            'rew': rewards[t].tolist()
        }
        if 'states' in data:
            step['state'] = data['states'][t].tolist()
        trajectories.append(step)

    worst_episode_data = {
        'metadata': {
            'env': env,
            'scenario': scenario,
            'quality': quality,
            'n_agents': n_agents,
            'n_timesteps': episode_length,
            'obs_dim': observations.shape[-1],
            'act_dim': actions.shape[-1],
            'note': f'Worst episode: steps {worst_episode_start}-{worst_episode_end}',
            'original_start_step': worst_episode_start,
            'cumulative_reward': float(episode_rewards[worst_indices[0]])
        },
        'trajectories': trajectories
    }

    worst_output_file = Path(npz_file).parent / f"{Path(npz_file).stem}_WORST.json"
    with open(worst_output_file, 'w') as f:
        json.dump(worst_episode_data, f, indent=2)

    print(f"   ✅ Saved: {worst_output_file.name}")

    print(f"\n💡 To visualize (from Windows):")
    print(f"   Best:  python verification/record_mamujoco.py \"{output_file.as_posix().replace('/mnt/c', 'C:')}\" best.mp4")
    print(f"   Worst: python verification/record_mamujoco.py \"{worst_output_file.as_posix().replace('/mnt/c', 'C:')}\" worst.mp4")

    return best_indices, worst_indices, episode_starts, episode_rewards


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_best_episodes_npz.py <path_to_npz_file> [episode_length] [top_n]")
        print("\nExample:")
        print("  python find_best_episodes_npz.py data.npz 1000 5")
        print("\nArguments:")
        print("  episode_length: Length of episode window (default: 1000)")
        print("  top_n: Number of top episodes to show (default: 5)")
        sys.exit(1)

    npz_file = sys.argv[1]
    episode_length = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    find_best_episodes(npz_file, episode_length, top_n)
