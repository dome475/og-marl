#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Record OG-MARL exported data as a video by replaying in MAMuJoCo
"""

import json
import numpy as np
from pathlib import Path
import sys
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def record_trajectory(json_file, output_video, num_steps=1000, fps=30):
    """
    Load JSON data and record video of MAMuJoCo environment

    Args:
        json_file: Path to exported JSON file
        output_video: Path to save video file (e.g., output.mp4)
        num_steps: Number of steps to record (default 1000)
        fps: Frames per second for output video (default 30)
    """

    # Load exported data
    print(f"📂 Loading {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    trajectories = data['trajectories']

    print(f"\n📊 Dataset Info:")
    print(f"   Environment: {metadata['env']}")
    print(f"   Scenario: {metadata['scenario']}")
    print(f"   Agents: {metadata['n_agents']}")
    print(f"   Recording {min(num_steps, len(trajectories))} steps")

    # Add og-marl to path
    og_marl_path = Path(__file__).parent.parent
    if str(og_marl_path) not in sys.path:
        sys.path.insert(0, str(og_marl_path))

    # Import and create environment with RGB array rendering
    print(f"\n🎮 Creating environment...")

    import gymnasium as gym
    from og_marl.wrapped_environments.gymnasium_mamujoco import get_env_config
    import gymnasium_robotics

    scenario = metadata['scenario']
    env_config = get_env_config(scenario)

    # Create environment with rgb_array rendering
    env = gymnasium_robotics.mamujoco_v1.parallel_env(**env_config, render_mode="rgb_array")

    print(f"   ✅ Environment created")

    # Reset environment
    obs, _ = env.reset()

    print(f"\n🎬 Recording video...")
    print(f"   This may take a while...")

    frames = []

    try:
        for i in range(min(num_steps, len(trajectories))):
            step_data = trajectories[i]

            # Get stored action
            stored_actions = np.array(step_data['act'])  # Shape: (n_agents, act_dim)
            stored_reward = np.array(step_data['rew'])

            # Format actions for environment
            actions = {f"agent_{j}": stored_actions[j] for j in range(metadata['n_agents'])}

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(actions)

            # Capture frame
            frame = env.render()
            frames.append(frame)

            # Print progress
            if i % 100 == 0:
                print(f"   Step {i}/{min(num_steps, len(trajectories))} | {len(frames)} frames captured")

            # Check termination
            done = any(terminated.values()) if isinstance(terminated, dict) else terminated
            trunc = any(truncated.values()) if isinstance(truncated, dict) else truncated

            if done or trunc:
                print(f"   Episode ended at step {i}, resetting...")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print(f"\n\n⏸️  Recording stopped by user")

    env.close()

    # Save video
    print(f"\n💾 Saving video to {output_video}...")

    try:
        import imageio

        # Convert frames to proper format
        frames_array = np.array(frames)
        print(f"   Video shape: {frames_array.shape} (frames, height, width, channels)")

        # Save as MP4
        imageio.mimsave(output_video, frames_array, fps=fps)

        file_size = Path(output_video).stat().st_size / (1024*1024)
        print(f"\n✅ Video saved! ({file_size:.1f} MB)")
        print(f"   Open with: {output_video}")

    except ImportError:
        print(f"   ❌ imageio not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imageio", "imageio-ffmpeg"])

        import imageio
        frames_array = np.array(frames)
        imageio.mimsave(output_video, frames_array, fps=fps)

        file_size = Path(output_video).stat().st_size / (1024*1024)
        print(f"\n✅ Video saved! ({file_size:.1f} MB)")
        print(f"   Open with: {output_video}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python record_mamujoco.py <path_to_json_file> [output_video] [num_steps] [fps]")
        print("\nExample:")
        print("  python record_mamujoco.py /path/to/data.json output.mp4 1000 30")
        print("\nArguments:")
        print("  output_video: Output video file path (default: visualization.mp4)")
        print("  num_steps: Number of steps to record (default: 1000)")
        print("  fps: Frames per second (default: 30)")
        sys.exit(1)

    json_file = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else "visualization.mp4"
    num_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    fps = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    record_trajectory(json_file, output_video, num_steps, fps)
