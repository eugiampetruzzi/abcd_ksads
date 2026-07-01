"""Tests for the derivatives validation/comparison tool."""

import hashlib
import json

import pandas as pd

from abcd_ksads import validate


def _frame():
    return pd.DataFrame(
        {"id": ["a", "b", "c"], "grp": ["x", "x", "y"], "val": [1.0, 2.0, 3.0]}
    )


# ---- load_table -------------------------------------------------------------


def test_load_table_csv_and_parquet(tmp_path):
    df = _frame()
    df.to_csv(tmp_path / "t.csv", index=False)
    df.to_parquet(tmp_path / "t.parquet", index=False)
    assert list(validate.load_table(tmp_path / "t.csv").columns) == ["id", "grp", "val"]
    assert len(validate.load_table(tmp_path / "t.parquet")) == 3


# ---- compare_tables ---------------------------------------------------------


def test_compare_tables_identical():
    r = validate.compare_tables(_frame(), _frame())
    assert r["aligned"] is True
    assert r["match_fraction"] == 1.0
    assert r["n_mismatch"] == 0
    assert r["max_abs_diff"] == 0.0
    assert r["cols_only_orig"] == [] and r["cols_only_new"] == []


def test_compare_tables_within_tolerance_matches():
    b = _frame()
    b.loc[0, "val"] = 1.0 + 1e-9  # below atol=1e-8
    r = validate.compare_tables(_frame(), b)
    assert r["n_mismatch"] == 0
    assert 0 < r["max_abs_diff"] < 1e-8


def test_compare_tables_large_diff_flags_mismatch():
    b = _frame()
    b.loc[0, "val"] = 99.0
    r = validate.compare_tables(_frame(), b)
    assert r["n_mismatch"] == 1
    assert r["match_fraction"] < 1.0
    assert r["per_column_mismatch"]["val"] == 1
    assert r["max_abs_diff"] == 98.0


def test_compare_tables_is_order_independent():
    b = _frame().iloc[[2, 0, 1]].reset_index(drop=True)  # shuffled rows
    r = validate.compare_tables(_frame(), b)
    assert r["match_fraction"] == 1.0
    assert r["n_mismatch"] == 0


def test_compare_tables_reports_extra_column():
    b = _frame()
    b["extra"] = 1
    r = validate.compare_tables(_frame(), b)
    assert r["cols_only_new"] == ["extra"]
    assert r["cols_only_orig"] == []


def test_compare_tables_row_count_differs_reports_overlap():
    b = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)  # 4 rows vs 3
    r = validate.compare_tables(_frame(), b)
    assert r["aligned"] is False
    assert r["n_rows_orig"] == 3 and r["n_rows_new"] == 4
    assert r["row_overlap_fraction"] is not None
    assert 0 < r["row_overlap_fraction"] <= 1.0


def test_compare_tables_nan_equal():
    a = pd.DataFrame({"id": ["a", "b"], "val": [1.0, float("nan")]})
    r = validate.compare_tables(a, a.copy())
    assert r["match_fraction"] == 1.0


# ---- participant_id hashing -------------------------------------------------


def test_hash_participant_id_matches_sha256():
    v = "sub-9N9B1Z7A"
    assert validate.hash_participant_id(v) == "sub-" + hashlib.sha256(v.encode()).hexdigest()[:8]


def test_compare_tables_hashes_reference_ids_to_align():
    a = pd.DataFrame({"participant_id": ["sub-AAAA", "sub-BBBB"], "val": [1.0, 2.0]})
    b = pd.DataFrame(
        {
            "participant_id": [validate.hash_participant_id("sub-AAAA"),
                               validate.hash_participant_id("sub-BBBB")],
            "val": [1.0, 2.0],
        }
    )
    r = validate.compare_tables(a, b)
    assert r["match_fraction"] == 1.0
    assert r["n_mismatch"] == 0
    assert r["hashed_id_cols"] == ["participant_id"]


def test_compare_tables_detects_value_diff_after_id_hash():
    a = pd.DataFrame({"participant_id": ["sub-AAAA", "sub-BBBB"], "val": [1.0, 2.0]})
    b = pd.DataFrame(
        {
            "participant_id": [validate.hash_participant_id("sub-AAAA"),
                               validate.hash_participant_id("sub-BBBB")],
            "val": [1.0, 9.0],
        }
    )
    r = validate.compare_tables(a, b)
    assert r["n_mismatch"] == 1


# ---- compare_json -----------------------------------------------------------


def test_compare_json_identical():
    obj = {"a": 1, "b": {"c": [1, 2, 3]}}
    r = validate.compare_json(obj, json.loads(json.dumps(obj)))
    assert r["match_fraction"] == 1.0
    assert r["n_mismatch"] == 0


def test_compare_json_reports_mismatched_paths():
    a = {"a": 1, "b": {"c": 2}}
    b = {"a": 1, "b": {"c": 5}}
    r = validate.compare_json(a, b)
    assert r["n_mismatch"] == 1
    assert any("c" in p for p in r["mismatched_paths"])


def test_compare_json_numeric_within_tolerance():
    a = {"x": 1.0}
    b = {"x": 1.0 + 1e-10}
    r = validate.compare_json(a, b)
    assert r["n_mismatch"] == 0


# ---- compare_files / directories --------------------------------------------


def test_compare_files_status_identical(tmp_path):
    _frame().to_csv(tmp_path / "o.csv", index=False)
    _frame().to_csv(tmp_path / "n.csv", index=False)
    r = validate.compare_files(tmp_path / "o.csv", tmp_path / "n.csv")
    assert r["status"] == "identical"
    assert r["kind"] == "table"


def test_compare_files_status_differs(tmp_path):
    _frame().to_csv(tmp_path / "o.csv", index=False)
    b = _frame()
    b.loc[0, "val"] = 99.0
    b.to_csv(tmp_path / "n.csv", index=False)
    r = validate.compare_files(tmp_path / "o.csv", tmp_path / "n.csv")
    assert r["status"] == "differs"


def test_compare_directories_flags_missing_and_present(tmp_path):
    orig = tmp_path / "orig"
    new = tmp_path / "new"
    orig.mkdir()
    new.mkdir()
    _frame().to_csv(orig / "a.csv", index=False)
    _frame().to_csv(orig / "b.csv", index=False)
    _frame().to_csv(new / "a.csv", index=False)  # b.csv missing in new
    results = {r["file"]: r for r in validate.compare_directories(orig, new)}
    assert results["a.csv"]["status"] == "identical"
    assert results["b.csv"]["status"] == "missing"


def test_report_dataframe_has_expected_columns(tmp_path):
    orig = tmp_path / "orig"
    new = tmp_path / "new"
    orig.mkdir()
    new.mkdir()
    _frame().to_csv(orig / "a.csv", index=False)
    _frame().to_csv(new / "a.csv", index=False)
    rep = validate.report_dataframe(validate.compare_directories(orig, new))
    for col in ("file", "kind", "status", "match_fraction"):
        assert col in rep.columns
