#!/usr/bin/env python3
"""
Windows-compatible OG-MARL vault to JSON converter
Fixes path issues on Windows systems
"""

import json
import numpy as np
from pathlib import Path
import os
import sys

# Try to import required packages
try:
    from flashbax.vault import Vault
    import jax
    FLASHBAX_AVAILABLE = True
except ImportError:
    FLASHBAX_AVAILABLE = False
    print("⚠️ Flashbax not installed. Install with: pip install flashbax jax")


def convert_vault_to_json(vault_path, quality="Replay"):
    """
    Convert vault data to readable JSON format - Windows compatible version
    """

    if not FLASHBAX_AVAILABLE:
        print("\nTo use this script, install Flashbax:")
        print("pip install flashbax jax")
        return None

    print(f"Loading vault: {vault_path}")
    print(f"Quality level: {quality}")

    # Convert to Path object and get absolute path
    vault_path = Path(vault_path).absolute()

    # METHOD 1: Try with forward slashes (Unix-style)
    try:
        vault_path_unix = vault_path.as_posix()
        print(f"Trying Unix-style path: {vault_path_unix}")

        vault = Vault(vault_path_unix, vault_uid=quality)
        experience = vault.read().experience
        print("✅ Successfully loaded with Unix-style path!")

    except Exception as e1:
        print(f"Unix-style path failed: {str(e1)[:100]}...")

        # METHOD 2: Try with relative path from current directory
        try:
            current_dir = Path.cwd()
            rel_path = vault_path.relative_to(current_dir)
            rel_path_str = rel_path.as_posix()
            print(f"\nTrying relative path: {rel_path_str}")

            vault = Vault(rel_path_str, vault_uid=quality)
            experience = vault.read().experience
            print("✅ Successfully loaded with relative path!")

        except Exception as e2:
            print(f"Relative path failed: {str(e2)[:100]}...")

            # METHOD 3: Try changing directory and using local path
            try:
                vault_dir = vault_path / quality
                if vault_dir.exists():
                    print(f"\nTrying direct directory access: {vault_dir}")

                    # Change to parent directory
                    original_dir = Path.cwd()
                    try:
                        os.chdir(vault_path.parent)

                        # Use just the folder name
                        vault_name = vault_path.name
                        vault = Vault(vault_name, vault_uid=quality)
                        experience = vault.read().experience

                        # Change back
                        os.chdir(original_dir)
                        print("✅ Successfully loaded with directory change!")
                    except Exception as e_inner:
                        # Make sure we restore directory even on error
                        os.chdir(original_dir)
                        raise e_inner
                else:
                    raise Exception(f"Vault directory not found: {vault_dir}")

            except Exception as e3:
                print(f"Directory method failed: {str(e3)[:100]}...")

                # METHOD 4: Manual loading attempt
                print("\n❌ All automatic methods failed. Trying manual approach...")
                return load_vault_manually(vault_path, quality)

    # If we got here, we successfully loaded the vault
    # Convert to numpy
    data = jax.tree.map(lambda x: np.array(x), experience)

    print("\n📊 Data structure:")
    for key in data.keys():
        if isinstance(data[key], dict):
            print(f"  {key}:")
            for subkey in data[key].keys():
                print(f"    {subkey}: {data[key][subkey].shape}")
        else:
            print(f"  {key}: {data[key].shape}")

    # Create readable format
    readable_data = extract_readable_data(data)

    return readable_data


def load_vault_manually(vault_path, quality):
    """
    Manual approach for loading vault data on Windows
    """
    vault_path = Path(vault_path)
    vault_dir = vault_path / quality

    print(f"Manual loading from: {vault_dir}")

    # Check if the directory exists
    if not vault_dir.exists():
        print(f"❌ Directory not found: {vault_dir}")
        print(f"Available directories: {list(vault_path.iterdir())}")
        return None

    # Try to read the manifest
    manifest_file = vault_dir / "manifest.ocdbt"
    if manifest_file.exists():
        print(f"✅ Found manifest: {manifest_file}")

    # Since manual loading is complex, provide instructions
    print("\n📝 Manual loading instructions:")
    print("1. Navigate to the vault directory in terminal/cmd:")
    print(f"   cd {vault_path.parent}")
    print("\n2. Run Python from that directory:")
    print("   python")
    print("\n3. Load the vault with relative path:")
    print(f"   from flashbax.vault import Vault")
    print(f"   vault = Vault('{vault_path.name}', vault_uid='{quality}')")
    print(f"   experience = vault.read().experience")

    return None


def extract_readable_data(data):
    """
    Extract readable data from the loaded vault
    """
    # Get dimensions
    batch_size = data['observations'].shape[0]
    n_timesteps = data['observations'].shape[1]
    n_agents = data['observations'].shape[2]
    obs_dim = data['observations'].shape[3]
    action_dim = data['actions'].shape[3]

    print(f"\n📈 Dataset info:")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Agents: {n_agents}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")

    # Create readable format
    readable_data = {
        "metadata": {
            "n_agents": int(n_agents),
            "n_timesteps": int(n_timesteps),
            "observation_dim": int(obs_dim),
            "action_dim": int(action_dim)
        },
        "trajectories": [],
        "statistics": {}
    }

    # Extract first 100 timesteps
    max_steps = min(100, n_timesteps)

    for t in range(max_steps):
        timestep = {
            "timestep": t,
            "agents": []
        }

        for agent_id in range(n_agents):
            agent_data = {
                "agent_id": agent_id,
                "observation": data['observations'][0, t, agent_id].tolist(),
                "action": data['actions'][0, t, agent_id].tolist(),
                "reward": float(data['rewards'][0, t, agent_id]),
                "terminal": bool(data['terminals'][0, t, agent_id]),
                "truncation": bool(data['truncations'][0, t, agent_id])
            }
            timestep["agents"].append(agent_data)

        # Add global state if available
        if 'infos' in data and 'state' in data['infos']:
            timestep["global_state"] = data['infos']['state'][0, t].tolist()

        readable_data["trajectories"].append(timestep)

    # Calculate statistics
    all_rewards = data['rewards'][0].flatten()
    readable_data["statistics"] = {
        "total_reward": float(np.sum(all_rewards)),
        "mean_reward": float(np.mean(all_rewards)),
        "min_reward": float(np.min(all_rewards)),
        "max_reward": float(np.max(all_rewards)),
        "std_reward": float(np.std(all_rewards))
    }

    # Count episodes
    episode_ends = np.logical_or(
        data['terminals'][0].any(axis=1),
        data['truncations'][0].any(axis=1)
    )
    readable_data["statistics"]["n_episodes"] = int(np.sum(episode_ends))

    return readable_data


def save_json(data, output_path):
    """Save data to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    file_size = Path(output_path).stat().st_size / 1024
    print(f"\n✅ Saved to: {output_path} ({file_size:.1f} KB)")


def windows_vault_workaround(vault_path, quality="Replay"):
    """
    Workaround specifically for Windows file path issues
    """
    print("\n🔧 Using Windows-specific workaround...")

    # Convert to absolute path
    vault_path = Path(vault_path).absolute()

    # Create a batch file to run the conversion
    batch_content = f"""
@echo off
cd /d "{vault_path.parent}"
python -c "
from flashbax.vault import Vault
import jax
import json
import numpy as np

vault = Vault('{vault_path.name}', vault_uid='{quality}')
experience = vault.read().experience
data = jax.tree.map(lambda x: np.array(x), experience)

# Extract first 10 timesteps as sample
sample = {{}}
for key in data.keys():
    if isinstance(data[key], dict):
        sample[key] = {{}}
        for subkey in data[key].keys():
            sample[key][subkey] = data[key][subkey][0, :10].tolist()
    else:
        sample[key] = data[key][0, :10].tolist()

# Save sample
with open('sample_output.json', 'w') as f:
    json.dump(sample, f, indent=2)

print('Sample saved to sample_output.json')
"
pause
"""

    batch_file = vault_path.parent / "convert_vault.bat"
    with open(batch_file, 'w') as f:
        f.write(batch_content)

    print(f"Created batch file: {batch_file}")
    print("Run this batch file to convert the vault data.")

    return batch_file


# Main execution
if __name__ == "__main__":
    print("🔄 OG-MARL Vault to JSON Converter (Windows Edition)")
    print("=" * 50)

    if len(sys.argv) > 1:
        vault_path = sys.argv[1]
        quality = sys.argv[2] if len(sys.argv) > 2 else "Replay"

        # Try to convert
        json_data = convert_vault_to_json(vault_path, quality)

        if json_data:
            # Save the data
            output_path = Path(vault_path).stem + "_readable.json"
            save_json(json_data, output_path)

            print(f"\n📊 Summary:")
            print(f"   Agents: {json_data['metadata']['n_agents']}")
            print(f"   Timesteps: {json_data['metadata']['n_timesteps']}")
            print(f"   Total reward: {json_data['statistics']['total_reward']:.2f}")
            print(f"   Episodes: {json_data['statistics']['n_episodes']}")
        else:
            print("\n🔧 Alternative approach:")
            print("Since automatic loading failed, try this workaround:")

            # Create batch file for Windows
            if sys.platform == "win32":
                batch_file = windows_vault_workaround(vault_path, quality)
                print(f"\n1. Run the batch file: {batch_file}")
                print("2. Or use the Python script approach below")

            print("\n📝 Python script approach:")
            print(f"1. Open command prompt/terminal")
            print(f"2. Navigate to: {Path(vault_path).parent}")
            print(f"3. Run this Python code:")
            print("-" * 40)
            print(f"""
from flashbax.vault import Vault
import jax, json, numpy as np

# Load vault
vault = Vault('{Path(vault_path).name}', vault_uid='{quality}')
experience = vault.read().experience
data = jax.tree.map(lambda x: np.array(x), experience)

# Convert to JSON (first 100 timesteps)
output = {{"metadata": {{}}, "trajectories": []}}
for t in range(min(100, data['observations'].shape[1])):
    step = {{
        "timestep": t,
        "observations": data['observations'][0, t].tolist(),
        "actions": data['actions'][0, t].tolist(),
        "rewards": data['rewards'][0, t].tolist()
    }}
    output["trajectories"].append(step)

# Save
with open('output.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Saved to output.json")
""")
            print("-" * 40)
    else:
        print("Usage: python script.py <vault_path> [quality]")
        print("Example: python script.py vaults/og_marl/gymnasium_mamujoco/2halfcheetah.vlt Replay")
