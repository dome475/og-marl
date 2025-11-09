#!/usr/bin/env python3
"""
Analyze OG-MARL data to find best performing episodes
"""

import json
import numpy as np
from pathlib import Path
import sys

def find_best_episodes(json_file, episode_length=1000, top_n=5):
    """
    Analyze dataset and find episodes with highest cumulative rewards

    Args:
        json_file: Path to exported JSON file
        episode_length: Length of episode window to analyze
        top_n: Number of top episodes to identify
    """

    print(f"📂 Loading {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    trajectories = data['trajectories']

    print(f"\n📊 Dataset Info:")
    print(f"   Environment: {metadata['env']}")
    print(f"   Scenario: {metadata['scenario']}")
    print(f"   Total timesteps: {len(trajectories)}")

    print(f"\n🔍 Analyzing trajectories in {episode_length}-step windows...")

    # Calculate cumulative rewards for sliding windows
    episode_rewards = []
    episode_starts = []

    for start in range(0, len(trajectories) - episode_length, episode_length // 2):  # 50% overlap
        episode_reward = sum(
            np.mean(trajectories[t]['rew'])
            for t in range(start, min(start + episode_length, len(trajectories)))
        )
        episode_rewards.append(episode_reward)
        episode_starts.append(start)

    episode_rewards = np.array(episode_rewards)

    # Find statistics
    print(f"\n📈 Reward Statistics (per {episode_length} steps):")
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
        print(f"   #{rank}: Steps {start_step:6d}-{start_step + episode_length:6d} | Total: {reward:7.2f} | Avg: {avg_reward:6.3f}")

    # Find worst episodes for comparison
    worst_indices = np.argsort(episode_rewards)[:min(top_n, len(episode_rewards))]

    print(f"\n😞 Worst {min(top_n, len(episode_rewards))} Episodes (for comparison):")
    for rank, idx in enumerate(worst_indices, 1):
        start_step = episode_starts[idx]
        reward = episode_rewards[idx]
        avg_reward = reward / episode_length
        print(f"   #{rank}: Steps {start_step:6d}-{start_step + episode_length:6d} | Total: {reward:7.2f} | Avg: {avg_reward:6.3f}")

    # Create a JSON with just the best episode
    best_episode_start = episode_starts[best_indices[0]]
    best_episode_end = min(best_episode_start + episode_length, len(trajectories))

    best_episode_data = {
        'metadata': metadata.copy(),
        'trajectories': trajectories[best_episode_start:best_episode_end]
    }
    best_episode_data['metadata']['note'] = f'Best episode: steps {best_episode_start}-{best_episode_end}'
    best_episode_data['metadata']['original_start_step'] = best_episode_start
    best_episode_data['metadata']['cumulative_reward'] = float(episode_rewards[best_indices[0]])

    output_file = Path(json_file).parent / f"{Path(json_file).stem}_BEST.json"
    with open(output_file, 'w') as f:
        json.dump(best_episode_data, f, indent=2)

    print(f"\n💾 Saved best episode to: {output_file.name}")

    # Also save worst for comparison
    worst_episode_start = episode_starts[worst_indices[0]]
    worst_episode_end = min(worst_episode_start + episode_length, len(trajectories))

    worst_episode_data = {
        'metadata': metadata.copy(),
        'trajectories': trajectories[worst_episode_start:worst_episode_end]
    }
    worst_episode_data['metadata']['note'] = f'Worst episode: steps {worst_episode_start}-{worst_episode_end}'
    worst_episode_data['metadata']['original_start_step'] = worst_episode_start
    worst_episode_data['metadata']['cumulative_reward'] = float(episode_rewards[worst_indices[0]])

    worst_output_file = Path(json_file).parent / f"{Path(json_file).stem}_WORST.json"
    with open(worst_output_file, 'w') as f:
        json.dump(worst_episode_data, f, indent=2)

    print(f"💾 Saved worst episode to: {worst_output_file.name}")

    print(f"\n💡 To visualize:")
    print(f"   Best:  python verification/record_mamujoco.py \"{output_file}\" best.mp4")
    print(f"   Worst: python verification/record_mamujoco.py \"{worst_output_file}\" worst.mp4")

    return best_indices, worst_indices, episode_starts, episode_rewards


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_best_episodes.py <path_to_json_file> [episode_length] [top_n]")
        print("\nExample:")
        print("  python find_best_episodes.py data.json 1000 5")
        print("\nArguments:")
        print("  episode_length: Length of episode window (default: 1000)")
        print("  top_n: Number of top episodes to show (default: 5)")
        sys.exit(1)

    json_file = sys.argv[1]
    episode_length = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    find_best_episodes(json_file, episode_length, top_n)
