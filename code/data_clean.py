# NRLW data cleaning entry point (meet requirements of Week-5).

# Generates:
#   - product/sets.csv   (model ready, one row per set)

# Optional:
#   - product/events_enriched.csv
#   - product/validation.json
#   - product/figures

import argparse
import json
from pathlib import Path
from NRLW_CORE import (
    ensure_outdir, load_raw_csv, classify_events,
    normalise_direction, build_set_table, validate_dataset, derive_half_from_halfTag
)

def main():
    parser = argparse.ArgumentParser(
        description="Build set-level CSV from raw NRLW event data."
    )
    parser.add_argument("--raw", required=True, help="Path to raw events CSV")
    parser.add_argument("--outdir", default="product", help="Output directory (default: product)")
    parser.add_argument("--no-events", action="store_true", help="Do not write events_enriched.csv")
    parser.add_argument("--no-validation", action="store_true", help="Do not write validation.json")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # 1) Load & tidy
    df = load_raw_csv(args.raw)
    df = derive_half_from_halfTag(df) 
    
    # 2) Direction normalisation
    df = normalise_direction(df)

    # 3) Event tagging (run/kick)
    df = classify_events(df)

    # 4) Aggregate to set-level
    sets = build_set_table(df)

    # 5) Save outputs
    (outdir / "sets.csv").parent.mkdir(parents=True, exist_ok=True)
    sets.to_csv(outdir / "sets.csv", index=False)

    if not args.no_events:
        df.to_csv(outdir / "events_enriched.csv", index=False)

    if not args.no_validation:
        report = validate_dataset(df, sets)
        with open(outdir / "validation.json", "w") as f:
            json.dump(report, f, indent=2)
        print("Validation:", report)

    print(f"[done] Wrote {outdir/'sets.csv'} (rows={len(sets):,})")

if __name__ == "__main__":
    main()
