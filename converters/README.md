# OG-MARL Data Converters

Convert OG-MARL vault data to usable formats.

## ⚠️ Important: Use WSL/Linux for Conversion

**The vault conversion requires WSL or native Linux.** Windows Python will fail due to tensorstore path issues.

**Setup:**
1. Open WSL terminal
2. Navigate to project: `cd /mnt/c/Users/dbehl/Documents/GitHub/og-marl`
3. Activate environment: `conda activate ogmarl`
4. Run converters using WSL paths (e.g., `/mnt/d/OG_MARL/...`)

**Inspection works on Windows**, but **conversion requires WSL**.

## Vault Structure

OG-MARL vaults are stored in this format:
```
scenario.vlt/
├── Quality_Name/              # e.g., "Replay", "0", "1", "Good", "Medium", "Poor"
│   ├── metadata.json          # Vault metadata
│   ├── manifest.ocdbt         # Data manifest
│   └── d/                     # Actual trajectory data
│       └── [tensorstore data files]
```

## Available Converters

### 1. Inspect Vault (`inspect_vault.py`)

List available qualities and metadata in a vault.

```bash
python converters/inspect_vault.py path/to/scenario.vlt
```

**Examples:**
```bash
# From flash drive (Windows)
python converters/inspect_vault.py "D:/og_marl_data/2halfcheetah.vlt"

# From flash drive (WSL)
python converters/inspect_vault.py /mnt/d/og_marl_data/2halfcheetah.vlt

# From local download
python converters/inspect_vault.py /mnt/c/Users/dbehl/Downloads/2halfcheetah.vlt
```

### 2. Vault to NumPy (`vault_to_npz.py`)

Convert vault data to compressed NumPy format (`.npz`). **Recommended for large datasets.**

```bash
python converters/vault_to_npz.py <vault_path> <output_dir> [--quality QUALITY] [--all-qualities]
```

**Arguments:**
- `vault_path`: Path to `.vlt` directory (can be on flash drive, downloads, etc.)
- `output_dir`: Where to save `.npz` files
- `--quality`: Specific quality to convert (default: auto-detect first available)
- `--all-qualities`: Convert all available qualities

**Examples:**
```bash
# From flash drive to local outputs
python converters/vault_to_npz.py "D:/og_marl_data/2halfcheetah.vlt" outputs/converted/

# WSL path (flash drive)
python converters/vault_to_npz.py /mnt/d/og_marl_data/2halfcheetah.vlt outputs/converted/

# Convert specific quality
python converters/vault_to_npz.py "D:/og_marl_data/2halfcheetah.vlt" outputs/converted/ --quality Replay

# Convert all qualities
python converters/vault_to_npz.py "D:/og_marl_data/2halfcheetah.vlt" outputs/converted/ --all-qualities
```

**Output format:**
- File: `{scenario}_{quality}.npz`
- Contains: observations, actions, rewards, states (if available), metadata
- Size: ~200-500 MB (compressed) vs GB for JSON

### 3. Vault to JSON (`vault_to_json.py`)

Convert vault data to JSON format. **Use for small samples only.**

```bash
python converters/vault_to_json.py <vault_path> <output_dir> [--quality QUALITY] [--max-timesteps N]
```

**Arguments:**
- `vault_path`: Path to `.vlt` directory
- `output_dir`: Where to save `.json` files
- `--quality`: Specific quality to convert (default: auto-detect)
- `--max-timesteps`: Limit exported timesteps (default: 10000)

**Examples:**
```bash
# From flash drive, first 10k timesteps
python converters/vault_to_json.py "D:/og_marl_data/2halfcheetah.vlt" outputs/converted/ --max-timesteps 10000

# WSL path, specific quality, 5k timesteps
python converters/vault_to_json.py /mnt/d/og_marl_data/2halfcheetah.vlt outputs/converted/ --quality Replay --max-timesteps 5000

# Quick test with 1k timesteps
python converters/vault_to_json.py "D:/og_marl_data/2halfcheetah.vlt" outputs/converted/ --max-timesteps 1000
```

## Workflow

1. **Download vault from HuggingFace** to your flash drive or local storage
   - Extract the `.zip` file to get the `.vlt` directory
2. **Inspect vault** to see available qualities
   ```bash
   python converters/inspect_vault.py "D:/og_marl_data/2halfcheetah.vlt"
   ```
3. **Convert to NPZ** for full dataset (recommended)
   ```bash
   python converters/vault_to_npz.py "D:/og_marl_data/2halfcheetah.vlt" outputs/converted/ --all-qualities
   ```
4. **Or convert to JSON** for small samples/visualization
   ```bash
   python converters/vault_to_json.py "D:/og_marl_data/2halfcheetah.vlt" outputs/converted/ --max-timesteps 1000
   ```

## Data Formats

### NPZ Format
```python
data = np.load('scenario_quality.npz')
observations = data['observations']  # (timesteps, agents, obs_dim)
actions = data['actions']            # (timesteps, agents, act_dim)
rewards = data['rewards']            # (timesteps, agents)
states = data['states']              # (timesteps, state_dim) - if available
```

### JSON Format
```json
{
  "metadata": {
    "env": "gymnasium_mamujoco",
    "scenario": "2halfcheetah",
    "quality": "Replay",
    "n_agents": 2,
    "n_timesteps": 10000,
    "obs_dim": 12,
    "act_dim": 3
  },
  "trajectories": [
    {
      "t": 0,
      "obs": [[...], [...]],
      "act": [[...], [...]],
      "rew": [...],
      "state": [...]
    }
  ]
}
```

## Requirements

```bash
pip install flashbax jax numpy
```
