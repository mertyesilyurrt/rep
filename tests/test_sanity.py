from pathlib import Path
import polars as pl


def test_trt_output_exists_and_has_columns():
    path = Path("data-clean/processed/trt_by_word.csv")
    assert path.exists(), f"Missing output file: {path}"

    df = pl.read_csv(path)

    required = {"subject_id", "stimulus", "condition", "index", "content", "total_reading_time"}
    missing = required.difference(df.columns)
    assert not missing, f"Missing columns: {missing}"

    assert df.height > 0, "No rows produced"

    # Null checks in key columns
    for c in ["subject_id", "stimulus", "content", "total_reading_time"]:
        assert df[c].null_count() == 0, f"Nulls found in {c}"

    # Basic value ranges
    trt_min = df["total_reading_time"].min()
    trt_max = df["total_reading_time"].max()
    assert trt_min >= 0, f"Negative TRT found: {trt_min}"
    # 4000 ms upper bound based on notebook's filtering step
    assert trt_max <= 4000, f"TRT exceeds 4000 ms upper bound: {trt_max}"

    # Allowed conditions
    conds = set(df["condition"].unique().to_list())
    allowed = {"neg", "pos", "zero"}
    unexpected = conds - allowed
    assert not unexpected, f"Unexpected condition values: {unexpected}"