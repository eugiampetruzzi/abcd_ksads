"""Tests for the raw-source ingest step (single consolidated wide table)."""

import json

import pandas as pd
import pytest

from abcd_ksads import ingest

# Three synthetic source tables written as exact text so faithful reading can be
# checked against known tokens (leading zeros, sentinel codes, empty cells). The
# leading "Unnamed: 0" column mimics the pandas-index artifact in the real files.
#
# table1 covers (sub-01/02/03, ses-00A); table2 covers sub-01 at two sessions;
# table3 is session-invariant (participant_id only, like ab_g_stc).
TABLE1_TSV = (
    "Unnamed: 0\tparticipant_id\tsession_id\tval\tcode\n"
    "0\tsub-01\tses-00A\t1\t007\n"
    "1\tsub-02\tses-00A\t555\t010\n"
    "2\tsub-03\tses-00A\t\t000\n"
)
TABLE1_JSON = {
    "MeasurementToolMetadata": {"Description": "ADHD parent"},
    "val": {"Description": "a value", "Levels": {"0": "No", "1": "Yes"}, "Derivative": False},
    "code": {"Description": "a code", "Derivative": False},
}

TABLE2_TSV = (
    "Unnamed: 0\tparticipant_id\tsession_id\tx\n"
    "0\tsub-01\tses-00A\t3.5\n"
    "1\tsub-01\tses-01A\t4.5\n"
)
TABLE2_JSON = {"x": {"Description": "x val", "Derivative": True}}

# session-invariant table: participant_id only, no session_id
TABLE3_TSV = (
    "Unnamed: 0\tparticipant_id\tfam\n"
    "0\tsub-01\tF1\n"
    "1\tsub-02\tF2\n"
)
TABLE3_JSON = {"fam": {"Description": "family id", "Derivative": False}}


@pytest.fixture
def pheno_dir(tmp_path):
    """A phenotype directory with three TSV files and their JSON sidecars."""
    (tmp_path / "mh_p_ksads__adhd.tsv").write_text(TABLE1_TSV)
    (tmp_path / "mh_p_ksads__adhd.json").write_text(json.dumps(TABLE1_JSON))
    (tmp_path / "ab_g_dyn.tsv").write_text(TABLE2_TSV)
    (tmp_path / "ab_g_dyn.json").write_text(json.dumps(TABLE2_JSON))
    (tmp_path / "ab_g_stc.tsv").write_text(TABLE3_TSV)
    (tmp_path / "ab_g_stc.json").write_text(json.dumps(TABLE3_JSON))
    return tmp_path


# ---- faithful reading -------------------------------------------------------


def test_discover_tsvs_finds_all_tsvs_sorted(pheno_dir):
    found = ingest.discover_tsvs(pheno_dir)
    assert [p.name for p in found] == [
        "ab_g_dyn.tsv",
        "ab_g_stc.tsv",
        "mh_p_ksads__adhd.tsv",
    ]


def test_read_tsv_faithful_drops_index_artifact(pheno_dir):
    df = ingest.read_tsv_faithful(pheno_dir / "mh_p_ksads__adhd.tsv")
    assert "Unnamed: 0" not in df.columns
    assert list(df.columns) == ["participant_id", "session_id", "val", "code"]


def test_read_tsv_faithful_preserves_values_as_strings(pheno_dir):
    df = ingest.read_tsv_faithful(pheno_dir / "mh_p_ksads__adhd.tsv")
    assert df["val"].tolist() == ["1", "555", ""]
    assert df["code"].tolist() == ["007", "010", "000"]
    assert df["val"].isna().sum() == 0


# ---- metadata dictionary ----------------------------------------------------


def test_load_sidecar_metadata_separates_tool_and_columns(pheno_dir):
    meta = ingest.load_sidecar_metadata(pheno_dir / "mh_p_ksads__adhd.json")
    assert meta["MeasurementToolMetadata"] == {"Description": "ADHD parent"}
    assert set(meta["columns"]) == {"val", "code"}
    assert meta["columns"]["val"]["Levels"] == {"0": "No", "1": "Yes"}


def test_load_sidecar_metadata_without_tool_metadata(pheno_dir):
    meta = ingest.load_sidecar_metadata(pheno_dir / "ab_g_dyn.json")
    assert meta["MeasurementToolMetadata"] == {}
    assert set(meta["columns"]) == {"x"}


def test_build_metadata_dictionary_keyed_by_table(pheno_dir):
    md = ingest.build_metadata_dictionary(pheno_dir)
    assert set(md) == {"mh_p_ksads__adhd", "ab_g_dyn", "ab_g_stc"}
    assert md["mh_p_ksads__adhd"]["columns"]["code"]["Description"] == "a code"


# ---- consolidation into one wide table --------------------------------------


def test_consolidate_one_row_per_participant_session(pheno_dir):
    wide = ingest.consolidate(pheno_dir)
    pairs = set(map(tuple, wide[["participant_id", "session_id"]].values))
    assert pairs == {
        ("sub-01", "ses-00A"),
        ("sub-02", "ses-00A"),
        ("sub-03", "ses-00A"),
        ("sub-01", "ses-01A"),
    }


def test_consolidate_has_all_variable_columns(pheno_dir):
    wide = ingest.consolidate(pheno_dir)
    assert {"participant_id", "session_id", "val", "code", "x", "fam"} == set(wide.columns)


def test_consolidate_preserves_values(pheno_dir):
    wide = ingest.consolidate(pheno_dir).set_index(["participant_id", "session_id"])
    assert wide.loc[("sub-01", "ses-00A"), "val"] == "1"
    assert wide.loc[("sub-02", "ses-00A"), "code"] == "010"
    assert wide.loc[("sub-01", "ses-01A"), "x"] == "4.5"


def test_consolidate_absent_rows_are_nan(pheno_dir):
    wide = ingest.consolidate(pheno_dir).set_index(["participant_id", "session_id"])
    # sub-01 at ses-01A is not in table1 -> its columns are missing (NaN)
    assert pd.isna(wide.loc[("sub-01", "ses-01A"), "val"])
    # sub-03 is not in table2 -> x missing
    assert pd.isna(wide.loc[("sub-03", "ses-00A"), "x"])


def test_consolidate_uses_category_dtype_for_variables(pheno_dir):
    # variable columns are dictionary-encoded (category) to keep the wide table small;
    # values are unchanged (checked by the preserve/absent/broadcast tests above)
    wide = ingest.consolidate(pheno_dir)
    for col in ("val", "code", "x", "fam"):
        assert isinstance(wide[col].dtype, pd.CategoricalDtype)


def test_consolidate_broadcasts_session_invariant_table(pheno_dir):
    wide = ingest.consolidate(pheno_dir).set_index(["participant_id", "session_id"])
    # fam comes from the session-less table, broadcast across sub-01's sessions
    assert wide.loc[("sub-01", "ses-00A"), "fam"] == "F1"
    assert wide.loc[("sub-01", "ses-01A"), "fam"] == "F1"
    assert wide.loc[("sub-02", "ses-00A"), "fam"] == "F2"
    # sub-03 absent from the session-less table -> NaN
    assert pd.isna(wide.loc[("sub-03", "ses-00A"), "fam"])


# ---- ingest orchestration ---------------------------------------------------


def test_ingest_writes_single_consolidated_parquet(pheno_dir, tmp_path):
    cache = tmp_path / "cache"
    ingest.ingest(pheno_dir, cache)
    assert (cache / "phenotype.parquet").is_file()
    # the per-file mirror is no longer produced
    assert not (cache / "mh_p_ksads__adhd.parquet").exists()


def test_ingest_creates_cache_dir_if_missing(pheno_dir, tmp_path):
    cache = tmp_path / "does_not_exist_yet"
    ingest.ingest(pheno_dir, cache)
    assert cache.is_dir()


def test_ingest_parquet_matches_consolidate(pheno_dir, tmp_path):
    cache = tmp_path / "cache"
    ingest.ingest(pheno_dir, cache)
    reloaded = pd.read_parquet(cache / "phenotype.parquet")
    expected = ingest.consolidate(pheno_dir)
    pd.testing.assert_frame_equal(reloaded, expected, check_dtype=False)


def test_ingest_writes_metadata_json(pheno_dir, tmp_path):
    cache = tmp_path / "cache"
    ingest.ingest(pheno_dir, cache)
    written = json.loads((cache / "metadata.json").read_text())
    assert written == ingest.build_metadata_dictionary(pheno_dir)


def test_ingest_returns_summary(pheno_dir, tmp_path):
    cache = tmp_path / "cache"
    summary = ingest.ingest(pheno_dir, cache)
    assert summary["n_tables"] == 3
    assert summary["n_rows"] == 4  # 4 unique (participant, session) pairs
    assert summary["n_columns"] == 4  # val, code, x, fam
