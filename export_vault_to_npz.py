#!/usr/bin/env python3
"""
Export OG-MARL vault data to efficient NumPy format
"""

import os
import sys
import shutil
from pathlib import Path
import zipfile
import numpy as np

# Configuration
ENVIRONMENTS = {
    'gymnasium_mamujoco': {
        'scenarios': ['2halfcheetah'],
        'folder': 'core/gymnasium_mamujoco',
        'qualities': ['Replay', 'Replay_Uniform_200episodes']
    }
}

def export_vault_to_npz(output_dir="/mnt/c/Users/dbehl/og_marl_data"):
    """Export vault to NumPy format (much more efficient than JSON)"""
    from huggingface_hub import hf_hub_download
    from flashbax.vault import Vault
    import jax

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path / "temp"
    temp_dir.mkdir(exist_ok=True)

    print(f"📁 Output directory: {output_path}")

    for env_name, config in ENVIRONMENTS.items():
        print(f"\n📦 Processing {env_name}")

        for scenario in config['scenarios']:
            zip_file = f"{config['folder']}/{scenario}.zip"
            print(f"  📥 Downloading {zip_file}...")

            try:
                local_zip = hf_hub_download(
                    repo_id="InstaDeepAI/og-marl",
                    repo_type="dataset",
                    filename=zip_file,
                    local_dir=str(temp_dir)
                )

                # Extract
                extract_dir = temp_dir / scenario
                extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(local_zip, 'r') as zf:
                    zf.extractall(extract_dir)

                # Find vault
                vault_dir = list(extract_dir.glob("**/*.vlt"))[0]
                print(f"  ✅ Using vault: {vault_dir}")

                # Get available qualities
                available_qualities = [d.name for d in vault_dir.iterdir() if d.is_dir()]
                print(f"  🔍 Available qualities: {available_qualities}")

                qualities_to_process = [q for q in config['qualities'] if q in available_qualities]
                if not qualities_to_process:
                    qualities_to_process = available_qualities

                for quality in qualities_to_process:
                    print(f"  🔄 Converting {scenario}/{quality}...")

                    # Load vault
                    vault = Vault(str(vault_dir), vault_uid=quality)
                    experience = vault.read().experience
                    data = jax.tree.map(lambda x: np.array(x), experience)

                    # Get shapes
                    n_timesteps = data['observations'].shape[1]
                    n_agents = data['observations'].shape[2]

                    print(f"    📊 {n_timesteps} timesteps, {n_agents} agents")
                    print(f"    💾 Saving to NumPy format...")

                    # Save as NPZ (compressed NumPy format)
                    output_file = output_path / f"{env_name}_{scenario}_{quality}.npz"

                    # Prepare data dictionary
                    save_dict = {
                        'observations': data['observations'][0],  # Remove batch dimension
                        'actions': data['actions'][0],
                        'rewards': data['rewards'][0],
                        'n_timesteps': n_timesteps,
                        'n_agents': n_agents,
                        'env': env_name,
                        'scenario': scenario,
                        'quality': quality,
                    }

                    # Add state if available
                    if 'infos' in data and 'state' in data['infos']:
                        save_dict['states'] = data['infos']['state'][0]

                    # Save compressed
                    np.savez_compressed(output_file, **save_dict)

                    size_mb = output_file.stat().st_size / (1024*1024)
                    print(f"    ✅ Saved {output_file.name} ({size_mb:.1f} MB)")

            except Exception as e:
                import traceback
                print(f"  ❌ Failed: {e}")
                traceback.print_exc()

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"\n✅ Done! Data saved to {output_path}")

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/dbehl/og_marl_data"

    print("📊 OG-MARL Vault to NumPy Exporter")
    print(f"Will save to: {output_dir}")

    export_vault_to_npz(output_dir)
