#!/usr/bin/env python3
"""
Download OG-MARL data from Parquet format (much more efficient!)
No vault conversion needed - works with pandas directly
"""

import sys
from pathlib import Path

def list_available_data():
    """List what's available in the Parquet dataset"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: Install datasets library:")
        print("  pip install datasets pyarrow pandas")
        sys.exit(1)

    print("Loading OG-MARL dataset info from Parquet...")
    print("(This loads metadata only, not the full data)")
    print()

    try:
        # Load dataset in streaming mode (no download yet)
        dataset = load_dataset(
            "InstaDeepAI/og-marl",
            split="train",
            streaming=True
        )

        # Get first few rows to see structure
        print("Dataset structure:")
        first_item = next(iter(dataset))

        print("\nAvailable columns:")
        for key in first_item.keys():
            value = first_item[key]
            if hasattr(value, 'shape'):
                print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  - {key}: {type(value).__name__}")

        print("\nFirst item sample:")
        for key, value in first_item.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")

        print("\n" + "="*50)
        print("To download specific scenarios:")
        print("  python converters/download_parquet.py --scenario 20_trains")
        print("\nTo list all scenarios:")
        print("  python converters/download_parquet.py --list-scenarios")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def download_scenario(scenario_name, output_dir, max_rows=None):
    """
    Download specific scenario from Parquet

    Args:
        scenario_name: Name of scenario (e.g., '20_trains', '2halfcheetah')
        output_dir: Where to save
        max_rows: Limit number of rows (None = all)
    """
    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        print("Error: Install dependencies:")
        print("  pip install datasets pyarrow pandas")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {scenario_name} from Parquet...")
    print(f"Output: {output_path}")
    if max_rows:
        print(f"Limiting to first {max_rows:,} rows")
    print()

    try:
        # Load dataset
        dataset = load_dataset(
            "InstaDeepAI/og-marl",
            split="train",
            streaming=max_rows is not None  # Stream if limiting rows
        )

        if max_rows:
            # Take first N rows
            rows = []
            for i, item in enumerate(dataset):
                if i >= max_rows:
                    break
                rows.append(item)
                if (i + 1) % 100 == 0:
                    print(f"  Downloaded {i+1:,} rows...")

            # Convert to DataFrame
            df = pd.DataFrame(rows)
        else:
            # Download all
            print("  Downloading full dataset...")
            dataset_dict = load_dataset("InstaDeepAI/og-marl", split="train")
            df = dataset_dict.to_pandas()

        # Save as parquet (compressed)
        output_file = output_path / f"{scenario_name}.parquet"
        df.to_parquet(output_file, compression='snappy')

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n✓ Saved: {output_file.name} ({size_mb:.1f} MB)")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")

        print("\nTo load:")
        print(f"  import pandas as pd")
        print(f"  df = pd.read_parquet('{output_file}')")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Download OG-MARL data from Parquet format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--list', action='store_true',
                       help='List available data')
    parser.add_argument('--scenario', help='Download specific scenario')
    parser.add_argument('--output-dir', default='outputs/parquet',
                       help='Output directory (default: outputs/parquet)')
    parser.add_argument('--max-rows', type=int,
                       help='Limit number of rows to download')

    args = parser.parse_args()

    if args.list or not args.scenario:
        list_available_data()
    else:
        download_scenario(args.scenario, args.output_dir, args.max_rows)
