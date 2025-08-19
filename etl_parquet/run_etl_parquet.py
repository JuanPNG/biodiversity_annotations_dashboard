from __future__ import annotations
import argparse
from pathlib import Path

# relative imports of sibling modules in the same package
from .prep_main_parquet import build_main_table
from .prep_biogeo import build_biogeo_long
from .prep_gbif import build_gbif_occurrences

def run() -> None:
    ap = argparse.ArgumentParser(description="Build all dashboard Parquets (main, biogeo_long, gbif_occurrences).")
    ap.add_argument("parquet_path", type=Path)
    ap.add_argument("--sample-rows", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # build
    main_df = build_main_table(str(args.parquet_path))
    bio_df  = build_biogeo_long(str(args.parquet_path))
    gbif_df = build_gbif_occurrences(str(args.parquet_path))

    # write
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
