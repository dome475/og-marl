"""Quick script to inspect action trajectories from the vault."""

from flashbax.vault import Vault
import jax
import numpy as np

# Load the vault
# Note: Using forward slashes for Windows compatibility
vault = Vault(
    rel_dir="vaults/og_marl/gymnasium_mamujoco",
    vault_name="2halfcheetah.vlt",
    vault_uid="Replay"  # or "Replay_Uniform_200episodes"
)

# Read the data
print("Loading data...")
all_data = vault.read()
offline_data = all_data.experience

# Convert to numpy
numpy_data = jax.tree_map(lambda x: np.array(x), offline_data)

print("\n=== Dataset Structure ===")
print(f"Actions shape: {numpy_data['actions'].shape}")
print(f"  - Batch dimension: {numpy_data['actions'].shape[0]}")
print(f"  - Timesteps: {numpy_data['actions'].shape[1]}")
print(f"  - Number of agents: {numpy_data['actions'].shape[2]}")
print(f"  - Action dimension per agent: {numpy_data['actions'].shape[3]}")

print(f"\nObservations shape: {numpy_data['observations'].shape}")
print(f"Rewards shape: {numpy_data['rewards'].shape}")
print(f"Terminals shape: {numpy_data['terminals'].shape}")

# Show a sample of joint actions
print("\n=== Sample Joint Actions (first 5 timesteps) ===")
for t in range(min(5, numpy_data['actions'].shape[1])):
    joint_action = numpy_data['actions'][0, t, :, :]  # [num_agents, action_dim]
    print(f"Timestep {t}:")
    for agent_id in range(joint_action.shape[0]):
        print(f"  Agent {agent_id}: {joint_action[agent_id]}")

# Find episode boundaries
print("\n=== Episode Information ===")
terminals = numpy_data['terminals'][0, :, 0]  # Get terminal flags
episode_ends = np.where(terminals == 1)[0]
print(f"Total timesteps: {len(terminals)}")
print(f"Number of episodes: {len(episode_ends)}")
if len(episode_ends) > 0:
    print(f"First episode length: {episode_ends[0] + 1}")
    print(f"Average episode length: {np.mean(np.diff(np.concatenate([[-1], episode_ends]))):.1f}")

print("\n=== Action Statistics ===")
all_actions = numpy_data['actions'][0]  # [timesteps, num_agents, action_dim]
print(f"Action range: [{all_actions.min():.3f}, {all_actions.max():.3f}]")
print(f"Action mean: {all_actions.mean():.3f}")
print(f"Action std: {all_actions.std():.3f}")

print("\nYou can access joint actions at timestep t as:")
print("  joint_action = numpy_data['actions'][0, t, :, :]")
print("  where joint_action.shape = [num_agents, action_dim]")
