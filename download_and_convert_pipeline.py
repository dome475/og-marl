#!/usr/bin/env python3
"""
OG-MARL Download & Convert for WSL
Simplified version that works correctly in Linux environment
"""

import os
import sys
import json
import shutil
from pathlib import Path
import zipfile

# Configuration
ENVIRONMENTS = {
    # 'smac_v2': {
    #     'scenarios': ['terran_5_vs_5_std_exp'],  # 7.52 MB - tested successfully
    #     'folder': 'core/smac_v2',
    #     'qualities': ['Replay']
    # },
    'gymnasium_mamujoco': {
        'scenarios': ['2halfcheetah'],  # 1.3 GB - continuous actions
        'folder': 'core/gymnasium_mamujoco',
        'qualities': ['Replay', 'Replay_Uniform_200episodes']  # Actual quality levels in vault
    }
}

# COMMENTED OUT - Add back after testing:
# 'smac_v1': {
#     'scenarios': ['3m', '2s3z'],  # WARNING: 3m is 1.39 GB, 2s3z is also large
#     'folder': 'core/smac_v1',
#     'qualities': ['Good', 'Medium', 'Poor']
# },
# 'rware': {
#     'scenarios': ['tiny-2ag'],  # 13.1 MB - second smallest option
#     'folder': 'prior_work/alberdice/rware',  # Note: folder path updated
#     'qualities': ['Expert']
# }

def wsl_to_windows_path(wsl_path):
    """Convert WSL path to Windows path if running Windows Python"""
    import platform

    # If we're running Windows Python, convert the path
    if platform.system() == "Windows":
        wsl_path = str(wsl_path)
        # Convert /mnt/d/... to D:\...
        if wsl_path.startswith("/mnt/"):
            drive_letter = wsl_path[5].upper()
            rest_of_path = wsl_path[6:].replace("/", "\\")
            return f"{drive_letter}:{rest_of_path}"
    return wsl_path

def download_and_convert(output_dir="/mnt/d/og_marl_data"):
    """Main function for WSL"""
    from huggingface_hub import hf_hub_download
    from flashbax.vault import Vault
    import jax
    import numpy as np
    import platform

    # Convert to Windows path if using Windows Python
    if platform.system() == "Windows":
        output_dir = wsl_to_windows_path(output_dir)
        print(f"🪟 Detected Windows Python - converted path to: {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path / "temp"
    temp_dir.mkdir(exist_ok=True)

    print(f"📁 Output directory: {output_path}")
    
    for env_name, config in ENVIRONMENTS.items():
        print(f"\n📦 Processing {env_name}")
        
        for scenario in config['scenarios']:
            # Download ZIP
            zip_file = f"{config['folder']}/{scenario}.zip"
            print(f"  📥 Downloading {zip_file}...")
            
            try:
                local_zip = hf_hub_download(
                    repo_id="InstaDeepAI/og-marl",
                    repo_type="dataset",
                    filename=zip_file,
                    local_dir=str(temp_dir)
                )
                print(f"  ✅ Downloaded to: {local_zip}")

                # Extract ZIP
                extract_dir = temp_dir / scenario
                extract_dir.mkdir(exist_ok=True)
                print(f"  📂 Extracting to: {extract_dir}")

                with zipfile.ZipFile(local_zip, 'r') as zf:
                    zf.extractall(extract_dir)
                    print(f"  📂 Extracted files: {zf.namelist()[:5]}...")  # Show first 5 files

                # Find vault directory
                print(f"  🔍 Looking for .vlt directories in {extract_dir}")
                vlt_dirs = list(extract_dir.glob("**/*.vlt"))
                print(f"  🔍 Found {len(vlt_dirs)} vault directories: {vlt_dirs}")

                if not vlt_dirs:
                    print(f"  ❌ No .vlt directory found!")
                    continue

                vault_dir = vlt_dirs[0]
                print(f"  ✅ Using vault: {vault_dir}")

                # Discover available qualities by listing subdirectories
                available_qualities = [d.name for d in vault_dir.iterdir() if d.is_dir()]
                print(f"  🔍 Available qualities in vault: {available_qualities}")

                # Process each quality (use available ones if config doesn't match)
                qualities_to_process = [q for q in config['qualities'] if q in available_qualities]

                # If none from config exist, use all available qualities
                if not qualities_to_process:
                    print(f"  ⚠️  None of the configured qualities {config['qualities']} found. Using available: {available_qualities}")
                    qualities_to_process = available_qualities

                for quality in qualities_to_process:
                    quality_path = vault_dir / quality
                    print(f"  🔍 Checking for quality '{quality}' at: {quality_path}")
                    print(f"  🔍 Exists: {quality_path.exists()}")

                    if quality_path.exists():
                        print(f"  🔄 Converting {scenario}/{quality}...")

                        # Load vault - works directly in Linux/WSL
                        vault = Vault(str(vault_dir), vault_uid=quality)
                        experience = vault.read().experience
                        data = jax.tree.map(lambda x: np.array(x), experience)

                        # Extract metadata
                        n_timesteps = data['observations'].shape[1]
                        n_agents = data['observations'].shape[2]

                        print(f"    📊 {n_timesteps} timesteps, {n_agents} agents")
                        print(f"    📊 Data shapes:")
                        print(f"       Observations: {data['observations'].shape}")
                        print(f"       Actions: {data['actions'].shape}")
                        print(f"       Rewards: {data['rewards'].shape}")

                        # Export ALL timesteps
                        max_t = n_timesteps
                        print(f"    💾 Exporting all {max_t} timesteps (this may take a while)...")
                        trajectories = []

                        for t in range(max_t):
                            step = {
                                't': t,
                                'obs': data['observations'][0, t].tolist(),
                                'act': data['actions'][0, t].tolist(),
                                'rew': data['rewards'][0, t].tolist()
                            }
                            if 'infos' in data and 'state' in data['infos']:
                                step['state'] = data['infos']['state'][0, t].tolist()
                            trajectories.append(step)

                        # Save JSON
                        output_file = output_path / f"{env_name}_{scenario}_{quality}.json"
                        output = {
                            'metadata': {
                                'env': env_name,
                                'scenario': scenario,
                                'quality': quality,
                                'n_agents': int(n_agents),
                                'n_timesteps': int(n_timesteps),
                                'obs_dim': int(data['observations'].shape[-1]),
                                'act_dim': int(data['actions'].shape[-1]) if 'actions' in data else None
                            },
                            'trajectories': trajectories
                        }

                        with open(output_file, 'w') as f:
                            json.dump(output, f, indent=2)

                        size_mb = output_file.stat().st_size / (1024*1024)
                        print(f"    ✅ Saved {output_file.name} ({size_mb:.1f} MB)")
                    else:
                        print(f"  ⚠️  Quality '{quality}' not found, skipping...")

            except Exception as e:
                import traceback
                print(f"  ❌ Failed: {e}")
                print(f"  📋 Traceback:")
                traceback.print_exc()
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n✅ Done! Data saved to {output_path}")

if __name__ == "__main__":
    # Get output directory with smart defaults
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        # Try multiple fallback locations
        candidates = [
            "/mnt/d/og_marl_data",  # D: drive (if available)
            "/mnt/c/Users/dbehl/og_marl_data",  # C: drive user directory
            str(Path.home() / "og_marl_data")  # Home directory
        ]

        output_dir = None
        for candidate in candidates:
            try:
                parent = Path(candidate).parent
                if parent.exists():
                    output_dir = candidate
                    break
            except:
                continue

        if output_dir is None:
            output_dir = str(Path.home() / "og_marl_data")
            print(f"⚠️  Could not find suitable drive, using home directory")

    print("📊 OG-MARL WSL Downloader")
    print(f"Will save to: {output_dir}")
    
    # Install requirements
    os.system("pip install -q huggingface-hub flashbax jax numpy")
    
    # Run
    download_and_convert(output_dir)