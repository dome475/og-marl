# OG-MARL Data Verification

Tools to verify exported OG-MARL dataset data by replaying in actual environments.

## Installation

```bash
# Install verification dependencies
pip install -r verification/requirements.txt

# For MAMuJoCo verification, also ensure you have MuJoCo installed
pip install mujoco gymnasium-robotics
```

## Usage

### Verify MAMuJoCo Data

Replay exported trajectory data in the actual MAMuJoCo environment:

```bash
# Basic verification (100 steps)
python verification/verify_mamujoco.py /path/to/exported_data.json

# Verify more steps
python verification/verify_mamujoco.py /path/to/exported_data.json 500

# With visualization
python verification/verify_mamujoco.py /path/to/exported_data.json 100 --render
```

### Example

```bash
python verification/verify_mamujoco.py \
    /mnt/c/Users/dbehl/og_marl_data/gymnasium_mamujoco_2halfcheetah_Replay.json \
    100
```

## What it checks

The verification script:
1. ✅ Loads the exported JSON data
2. ✅ Creates the corresponding environment
3. ✅ Replays the stored actions
4. ✅ Compares replayed observations with stored observations
5. ✅ Compares replayed rewards with stored rewards
6. ✅ Reports error statistics

## Expected Results

- **Mean error < 0.01**: Data is correct ✅
- **Mean error < 0.1**: Small numerical differences (acceptable) ⚠️
- **Mean error > 0.1**: Data mismatch - investigate! ❌

## Troubleshooting

**Import errors:**
```bash
pip install gymnasium-robotics mujoco
```

**Environment not found:**
- Check that scenario name matches OG-MARL conventions
- Verify the wrapped_environments module is available
