from flashbax.vault import Vault
import jax, json, numpy as np

print("Loading vault from project root...")

# Run from project root, use relative path from there
vault = Vault(
    rel_dir="vaults/og_marl/gymnasium_mamujoco",
    vault_name="2halfcheetah.vlt",
    vault_uid="Replay"
)

print("Reading experience data...")
experience = vault.read().experience
data = jax.tree.map(lambda x: np.array(x), experience)

print("\nData structure loaded:")
for key in data.keys():
    if isinstance(data[key], dict):
        print(f"  {key}:")
        for subkey in data[key].keys():
            print(f"    {subkey}: {data[key][subkey].shape}")
    else:
        print(f"  {key}: {data[key].shape}")

# Convert to JSON (first 100 timesteps)
print("\nConverting to JSON...")
trajectories = []
n_timesteps = min(100, data['observations'].shape[1])

for t in range(n_timesteps):
    step = {
        "timestep": t,
        "observations": data['observations'][0, t].tolist(),
        "actions": data['actions'][0, t].tolist(),
        "rewards": data['rewards'][0, t].tolist(),
        "terminals": data['terminals'][0, t].tolist(),
        "truncations": data['truncations'][0, t].tolist()
    }
    # Add state if available
    if 'infos' in data and 'state' in data['infos']:
        step["state"] = data['infos']['state'][0, t].tolist()
    trajectories.append(step)

# Create output with metadata
output = {
    "metadata": {
        "n_timesteps": int(data['observations'].shape[1]),
        "n_agents": int(data['observations'].shape[2]),
        "obs_dim": int(data['observations'].shape[3]),
        "action_dim": int(data['actions'].shape[3])
    },
    "trajectories": trajectories
}

# Save
with open('vault_output.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved to vault_output.json ({n_timesteps} timesteps)")
print(f"   Agents: {output['metadata']['n_agents']}")
print(f"   Total timesteps in vault: {output['metadata']['n_timesteps']}")
