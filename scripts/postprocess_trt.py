from __future__ import annotations
from pathlib import Path
import sys

try:
    import polars as pl
except Exception as e:
    print(f"[postprocess_trt] Failed to import polars: {e}")
    sys.exit(1)

OUT_PATH = Path("data-clean/processed/trt_by_word.csv")


def main() -> int:
    if not OUT_PATH.exists():
        print(f"[postprocess_trt] Skipping: missing {OUT_PATH}")
        return 0
    try:
        df = pl.read_csv(OUT_PATH)
    except Exception as e:
        print(f"[postprocess_trt] Could not read {OUT_PATH}: {e}")
        return 1

    if "total_reading_time" not in df.columns:
        print("[postprocess_trt] Missing 'total_reading_time' column; nothing to filter")
        return 0

    before = df.height
    df = df.filter(
        (pl.col("total_reading_time") >= 150) & (pl.col("total_reading_time") <= 4000)
    )
    after = df.height
    
    try:
        df.write_csv(OUT_PATH)
        print(f"[postprocess_trt] Filtered TRT: {before} -> {after} rows (removed {before - after})")
        return 0
    except Exception as e:
        print(f"[postprocess_trt] Could not write filtered data to {OUT_PATH}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())