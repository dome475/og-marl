#!/usr/bin/env python3
"""
Visualize OG-MARL exported data by replaying in MAMuJoCo with rendering
"""

import json
import numpy as np
from pathlib import Path
import sys
import time

def visualize_trajectory(json_file, num_steps=1000, speed=1.0):
    """
    Load JSON data and visualize in MAMuJoCo environment

    Args:
        json_file: Path to exported JSON file
        num_steps: Number of steps to visualize (default 1000)
        speed: Playback speed multiplier (default 1.0, use 0.5 for slower)
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
    print(f"   Visualizing {min(num_steps, len(trajectories))} steps")

    # Add og-marl to path
    og_marl_path = Path(__file__).parent.parent
    if str(og_marl_path) not in sys.path:
        sys.path.insert(0, str(og_marl_path))

    # Import and create environment with rendering
    from og_marl.wrapped_environments.gymnasium_mamujoco import GymnasiumMAMuJoCo

    print(f"\n🎮 Creating environment with rendering...")
    scenario = metadata['scenario']

    try:
        # Create base environment (not wrapped, so we can control rendering)
        env = GymnasiumMAMuJoCo(scenario=scenario, seed=42)

        # Try to enable rendering
        env.environment.unwrapped.render_mode = "human"
        print(f"   ✅ Environment created with visualization")
    except Exception as e:
        print(f"   ❌ Failed to create environment: {e}")
        return

    # Reset environment
    obs, info = env.reset()

    print(f"\n🎬 Starting visualization...")
    print(f"   Press Ctrl+C to stop")
    print(f"   Playback speed: {speed}x")

    try:
        for i in range(min(num_steps, len(trajectories))):
            step_data = trajectories[i]

            # Get stored action
            stored_actions = np.array(step_data['act'])  # Shape: (n_agents, act_dim)
            stored_reward = np.array(step_data['rew'])

            # Format actions for environment
            actions = {f"agent_{j}": stored_actions[j] for j in range(metadata['n_agents'])}

            # Step environment
            obs, reward, terminated, truncated, info = env.step(actions)

            # Render
            try:
                env.environment.render()
            except:
                pass  # Some environments auto-render

            # Print progress
            if i % 50 == 0:
                print(f"   Step {i}/{min(num_steps, len(trajectories))} | Stored reward: {np.mean(stored_reward):.3f}")

            # Control playback speed (MuJoCo timestep is typically 0.002s)
            time.sleep(0.002 / speed)

            # Check termination
            done = any(terminated.values()) if isinstance(terminated, dict) else terminated
            trunc = any(truncated.values()) if isinstance(truncated, dict) else truncated

            if done or trunc:
                print(f"\n   Episode ended at step {i}, resetting...")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print(f"\n\n⏸️  Visualization stopped by user")

    print(f"\n✅ Visualization complete!")
    env.environment.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_mamujoco.py <path_to_json_file> [num_steps] [speed]")
        print("\nExample:")
        print("  python visualize_mamujoco.py /path/to/data.json 1000 1.0")
        print("\nArguments:")
        print("  num_steps: Number of steps to visualize (default: 1000)")
        print("  speed: Playback speed multiplier (default: 1.0, use 0.5 for half speed)")
        sys.exit(1)

    json_file = sys.argv[1]
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    speed = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    visualize_trajectory(json_file, num_steps, speed)
