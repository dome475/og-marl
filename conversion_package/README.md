# OG-MARL Vault Conversion Package

This package contains everything needed to convert OG-MARL vault files to efficient NumPy format (.npz).

## ⚠️ IMPORTANT: Must Use WSL or Linux

**DO NOT run these scripts from Windows Command Prompt or PowerShell!**

The conversion **MUST** be run from:
- ✅ **WSL (Windows Subsystem for Linux)** - Recommended for Windows users
- ✅ **Native Linux** - Ubuntu, etc.
- ❌ **NOT Windows CMD/PowerShell** - Will fail with path errors

### How to Use WSL on Windows

1. **Install WSL** (if not already installed):
   - Open PowerShell as Administrator
   - Run: `wsl --install`
   - Restart computer
   - Default: Ubuntu Linux

2. **Open WSL Terminal:**
   - Press Windows key
   - Type "WSL" or "Ubuntu"
   - Open the WSL terminal

3. **Navigate to the package:**
   ```bash
   # If package is in Downloads:
   cd /mnt/c/Users/YourUsername/Downloads/conversion_package

   # If package is in Documents:
   cd /mnt/c/Users/YourUsername/Documents/conversion_package
   ```

4. **Run all commands from this WSL terminal**

## Requirements

- **WSL or Linux** (required!)
- **Python 3.8+** (preferably 3.10)
- **16GB+ RAM recommended** (32GB for very large vaults)

## Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `flashbax` - Vault reading library
- `jax` - Numerical computing
- `numpy` - Array operations

**Note:** Installation may take a few minutes and download ~500MB of packages.

### Step 2: Verify Installation

```bash
python -c "import flashbax; import jax; import numpy; print('✓ All dependencies installed')"
```

## Usage

### Quick Start

1. **Download a vault from HuggingFace:**
   - Go to: https://huggingface.co/datasets/InstaDeepAI/og-marl
   - Navigate to the vault you want (e.g., `core/flatland/20_trains.zip`)
   - Download and extract the `.zip` file
   - You'll get a folder like `20_trains.vlt/`

2. **Inspect the vault** (optional but recommended):
   ```bash
   python inspect_vault.py path/to/20_trains.vlt
   ```

   This shows:
   - Available quality levels
   - Number of timesteps
   - Number of agents
   - Data dimensions

3. **Convert to NPZ:**
   ```bash
   python vault_to_npz.py path/to/20_trains.vlt output/
   ```

   This creates: `output/20_trains_Medium.npz` (or similar)

### Examples

**Convert a vault with all quality levels:**
```bash
python vault_to_npz.py 20_trains.vlt output/ --all-qualities
```

**Convert specific quality only:**
```bash
python vault_to_npz.py 20_trains.vlt output/ --quality Medium
```

**Inspect before converting:**
```bash
# First check what's in the vault
python inspect_vault.py 20_trains.vlt

# Then convert
python vault_to_npz.py 20_trains.vlt output/
```

## What Gets Created

**Input:** `20_trains.vlt/` folder (vault format, ~2-5GB)
**Output:** `20_trains_Medium.npz` file (compressed NumPy, ~200-500MB)

The `.npz` file contains:
- `observations` - Agent observations at each timestep
- `actions` - Actions taken by agents
- `rewards` - Rewards received
- `states` - Global state (if available)
- Metadata (timesteps, agents, dimensions, etc.)

## Sending the Converted File

### Important: Only Send the .npz File!

**✅ Send:** `20_trains_Medium.npz` (~200-500MB)
**❌ Don't send:** The original `.vlt` vault folder (too large)

### How to Send

**Option 1: Google Drive**
1. Upload the `.npz` file to Google Drive
2. Right-click → Get shareable link
3. Send link to recipient

**Option 2: WeTransfer** (easiest, no account needed)
1. Go to: https://wetransfer.com
2. Upload the `.npz` file
3. Enter recipient's email
4. Click Transfer
5. Recipient gets download link via email

**Option 3: Dropbox / OneDrive**
1. Upload to your cloud storage
2. Create share link
3. Send to recipient

## Troubleshooting

### "Out of Memory" or "Aborted (core dumped)"
- Vault is too large for available RAM
- Try a smaller vault first
- Need more RAM (16GB+ recommended)

### "Vault does not exist" Error
- Check path is correct: `ls path/to/vault.vlt/`
- Ensure you extracted the .zip file
- Path should point to the `.vlt` folder, not a quality subfolder

### "Error parsing object member kvstore"
- You're using Windows Python (not supported)
- **Must use Linux or WSL** (Windows Subsystem for Linux)

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade flashbax jax numpy
```

## Expected Performance

| Vault Size | RAM Needed | Conversion Time | Output Size |
|------------|------------|-----------------|-------------|
| Small (2 agents, 100k steps) | 4-8 GB | 1-5 min | 50-100 MB |
| Medium (5 agents, 100k steps) | 8-16 GB | 5-15 min | 200-300 MB |
| Large (20 agents, 50k steps) | 16-32 GB | 10-30 min | 300-500 MB |

## Batch Processing

To convert multiple vaults:

```bash
# Create a simple script
for vault in *.vlt; do
    python vault_to_npz.py "$vault" output/
done
```

Or convert them one by one to monitor progress.

## Questions?

- Check vault structure: `python inspect_vault.py vault.vlt`
- Verify dependencies: `pip list | grep -E "flashbax|jax|numpy"`
- Check available disk space: `df -h`
- Monitor RAM usage: `top` or `htop`

## File Descriptions

- **`inspect_vault.py`** - Shows vault contents and metadata
- **`vault_to_npz.py`** - Converts vault to NumPy format
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This file

## Quick Reference: Complete Workflow

**All commands run from WSL terminal:**

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Download vault from HuggingFace (in browser)
# https://huggingface.co/datasets/InstaDeepAI/og-marl
# Save to: /mnt/c/Users/YourName/Downloads/

# 3. Extract the zip (in WSL)
cd /mnt/c/Users/YourName/Downloads/
unzip 20_trains.zip

# 4. Inspect vault (optional)
python /path/to/conversion_package/inspect_vault.py 20_trains.vlt

# 5. Convert
python /path/to/conversion_package/vault_to_npz.py 20_trains.vlt output/

# 6. Result is in output/20_trains_Medium.npz
# Upload this file to Google Drive/WeTransfer
```

## Summary Checklist

1. ✅ **Open WSL terminal** (NOT Windows CMD!)
2. ✅ Install dependencies: `pip install -r requirements.txt`
3. ✅ Download vault from HuggingFace and extract
4. ✅ Inspect (optional): `python inspect_vault.py vault.vlt`
5. ✅ Convert: `python vault_to_npz.py vault.vlt output/`
6. ✅ Send **only the .npz file** via cloud storage

**Remember: All commands must run in WSL terminal!**

Good luck! 🚀
