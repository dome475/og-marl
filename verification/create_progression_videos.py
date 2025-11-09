#!/usr/bin/env python3
"""
Create videos from different parts of the training run to show progression
"""

import json
import numpy as np
from pathlib import Path
import sys
import subprocess

def create_progression_videos(npz_file, num_videos=10, episode_length=500):
    """
    Sample different parts of the dataset and create videos

    Args:
        npz_file: Path to NPZ file (use Windows path: C:/...)
        num_videos: Number of videos to create (default: 10)
        episode_length: Length of each video segment (default: 500 steps)
    """

    # Handle both Windows and WSL paths
    npz_path = Path(npz_file)
    if not npz_path.exists():
        # Try converting WSL path to Windows path
        npz_file_str = str(npz_file).replace('/mnt/c', 'C:').replace('/mnt/d', 'D:').replace('/', '\\')
        npz_path = Path(npz_file_str)

    print(f"📂 Loading {npz_path}")
    data = np.load(npz_path)

    # Extract metadata
    env = str(data['env'])
    scenario = str(data['scenario'])
    quality = str(data['quality'])
    n_timesteps = int(data['n_timesteps'])
    n_agents = int(data['n_agents'])

    print(f"\n📊 Dataset Info:")
    print(f"   Environment: {env}")
    print(f"   Scenario: {scenario}")
    print(f"   Total timesteps: {n_timesteps}")
    print(f"   Creating {num_videos} videos of {episode_length} steps each")

    # Load trajectory data
    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']

    # Sample evenly across the dataset
    step_size = (n_timesteps - episode_length) // (num_videos - 1)
    sample_starts = [i * step_size for i in range(num_videos - 1)]
    sample_starts.append(n_timesteps - episode_length)  # Add last segment

    print(f"\n🎬 Creating {num_videos} video segments...")

    output_dir = npz_path.parent
    json_files = []

    for i, start in enumerate(sample_starts):
        end = start + episode_length

        # Calculate stats for this segment
        segment_reward = np.sum(rewards[start:end])
        avg_reward = segment_reward / episode_length

        print(f"\n   Video {i+1}/{num_videos}:")
        print(f"      Steps: {start:7d}-{end:7d}")
        print(f"      Cumulative reward: {segment_reward:8.2f}")
        print(f"      Avg reward/step: {avg_reward:6.3f}")

        # Export as JSON
        trajectories = []
        for t in range(start, end):
            step = {
                't': int(t - start),
                'obs': observations[t].tolist(),
                'act': actions[t].tolist(),
                'rew': rewards[t].tolist()
            }
            if 'states' in data:
                step['state'] = data['states'][t].tolist()
            trajectories.append(step)

        segment_data = {
            'metadata': {
                'env': env,
                'scenario': scenario,
                'quality': quality,
                'n_agents': n_agents,
                'n_timesteps': episode_length,
                'obs_dim': observations.shape[-1],
                'act_dim': actions.shape[-1],
                'note': f'Segment {i+1}/{num_videos}: steps {start}-{end}',
                'original_start_step': start,
                'cumulative_reward': float(segment_reward),
                'avg_reward': float(avg_reward)
            },
            'trajectories': trajectories
        }

        json_file = output_dir / f"{env}_{scenario}_{quality}_segment{i+1:02d}.json"
        with open(json_file, 'w') as f:
            json.dump(segment_data, f, indent=2)

        json_files.append(json_file)
        print(f"      ✅ Saved: {json_file.name}")

    # Generate videos
    print(f"\n🎥 Generating videos...")
    print(f"   This will take several minutes...")

    script_dir = Path(__file__).parent
    record_script = script_dir / "record_mamujoco.py"

    for i, json_file in enumerate(json_files):
        video_file = json_file.with_suffix('.mp4')

        print(f"\n   [{i+1}/{num_videos}] Recording {video_file.name}...")

        try:
            # Run record script
            cmd = [
                sys.executable,
                str(record_script),
                str(json_file),
                str(video_file),
                str(episode_length),
                "30"  # fps
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"         ✅ Video created!")
            else:
                print(f"         ❌ Failed: {result.stderr}")

        except Exception as e:
            print(f"         ❌ Error: {e}")

    print(f"\n✅ Done! Created {num_videos} videos in:")
    print(f"   {output_dir}")

    # Show summary
    print(f"\n📈 Progression Summary:")
    for i, json_file in enumerate(json_files):
        with open(json_file, 'r') as f:
            meta = json.load(f)['metadata']
        print(f"   Video {i+1}: Steps {meta['original_start_step']:7d}-{meta['original_start_step']+episode_length:7d} | "
              f"Avg reward: {meta['avg_reward']:6.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_progression_videos.py <path_to_npz_file> [num_videos] [episode_length]")
        print("\nExample:")
        print("  python create_progression_videos.py C:/Users/dbehl/og_marl_data/gymnasium_mamujoco_2halfcheetah_Replay.npz 10 500")
        print("\nArguments:")
        print("  num_videos: Number of videos to create (default: 10)")
        print("  episode_length: Steps per video (default: 500)")
        sys.exit(1)

    npz_file = sys.argv[1]
    num_videos = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    episode_length = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    create_progression_videos(npz_file, num_videos, episode_length)
