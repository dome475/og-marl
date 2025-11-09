#!/usr/bin/env python3
"""
Quick check of what's in the Parquet version
"""

try:
    from datasets import load_dataset_builder

    print("Checking OG-MARL Parquet structure...")

    # Get dataset info without downloading
    builder = load_dataset_builder("InstaDeepAI/og-marl")

    print("\nDataset info:")
    print(f"  Description: {builder.info.description}")
    print(f"  Features: {builder.info.features}")
    print(f"  Splits: {builder.info.splits}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
