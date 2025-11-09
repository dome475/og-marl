#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch record videos from existing JSON files
"""

import sys
import subprocess
from pathlib import Path

def batch_record(json_pattern, fps=30):
    """
    Record videos for all matching JSON files

    Args:
        json_pattern: Pattern to match JSON files (e.g., "*_segment*.json")
        fps: Frames per second
    """

    # Find all matching JSON files
    if '*' in json_pattern or '?' in json_pattern:
        # It's a glob pattern
        parent_dir = Path(json_pattern).parent
        pattern = Path(json_pattern).name
        json_files = sorted(parent_dir.glob(pattern))
    else:
        # It's a directory
        json_files = sorted(Path(json_pattern).glob("*_segment*.json"))

    if not json_files:
        print(f"No JSON files found matching: {json_pattern}")
        return

    print(f"Found {len(json_files)} JSON files to process")

    script_dir = Path(__file__).parent
    record_script = script_dir / "record_mamujoco.py"

    for i, json_file in enumerate(json_files, 1):
        video_file = json_file.with_suffix('.mp4')

        if video_file.exists():
            print(f"[{i}/{len(json_files)}] Skipping {video_file.name} (already exists)")
            continue

        print(f"[{i}/{len(json_files)}] Recording {video_file.name}...")

        try:
            # Get num_steps from JSON metadata
            import json
            with open(json_file, 'r') as f:
                metadata = json.load(f)['metadata']
                num_steps = metadata['n_timesteps']

            # Run record script
            cmd = [
                sys.executable,
                str(record_script),
                str(json_file),
                str(video_file),
                str(num_steps),
                str(fps)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                size_mb = video_file.stat().st_size / (1024*1024)
                print(f"    ✅ Video created ({size_mb:.1f} MB)")
            else:
                print(f"    ❌ Failed!")
                if result.stderr:
                    print(f"    Error: {result.stderr[:200]}")

        except Exception as e:
            print(f"    ❌ Error: {e}")

    print(f"\n✅ Done! Processed {len(json_files)} files")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_record_videos.py <json_pattern> [fps]")
        print("\nExamples:")
        print("  python batch_record_videos.py \"C:/Users/dbehl/og_marl_data/*_segment*.json\" 30")
        print("  python batch_record_videos.py \"C:/Users/dbehl/og_marl_data\" 30")
        sys.exit(1)

    json_pattern = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    batch_record(json_pattern, fps)
