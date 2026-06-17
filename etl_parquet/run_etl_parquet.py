"""Command-line entrypoint for building dashboard-ready Parquet files.

This script reads one integrated source Parquet and writes the processed files
consumed by the Dash app. By default it writes to data/processed, so use
--out-dir for review/versioned runs when working with production data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .prep_main_parquet import build_main_table
from .prep_biogeo import build_biogeo_long
from .prep_gbif import build_gbif_occurrences


def run() -> None:
    """Parse CLI arguments, build all processed tables, and write them to disk."""
    ap = argparse.ArgumentParser(description=(
        "Build dashboard-ready Parquet files: dashboard_main, "
        "biogeo_long, and gbif_occurrences."
    ))
    ap.add_argument("parquet_path", type=Path, help="Path to the integrated source Parquet file.")
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"), help=(
        "Directory for generated Parquet files. Defaults to data/processed, "
        "which is the app's active data directory."
    ))
    args = ap.parse_args()

    if not args.parquet_path.exists():
        raise FileNotFoundError(f"Source Parquet not found: {args.parquet_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build in memory before writing any outputs.
    main_df = build_main_table(str(args.parquet_path))
    bio_df  = build_biogeo_long(str(args.parquet_path))
    gbif_df = build_gbif_occurrences(str(args.parquet_path))

    # Write the fixed filenames expected by the dashboard.
    p_main = args.out_dir / "dashboard_main.parquet"
    p_bio  = args.out_dir / "biogeo_long.parquet"
    p_gbif = args.out_dir / "gbif_occurrences.parquet"

    main_df.to_parquet(p_main, index=False)
    bio_df.to_parquet(p_bio, index=False)
    gbif_df.to_parquet(p_gbif, index=False)

    print("\nWrote files:")
    print(f"  main         -> {p_main}")
    print(f"  biogeo_long  -> {p_bio}")
    print(f"  gbif_occs    -> {p_gbif}")

if __name__ == "__main__":
    run()
