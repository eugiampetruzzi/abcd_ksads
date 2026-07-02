"""Compare pipeline derivatives against a reference set and score their consistency.

Each file in a reference directory is compared to the same-named file produced by this
pipeline. Tables (csv/parquet) are compared order-independently with a numeric
tolerance; JSON is compared leaf-by-leaf; other binaries by byte equality. Results
carry a per-file status and consistency metrics.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

RTOL = 1e-5
ATOL = 1e-8
HASH_ID_COLS = ("participant_id",)

_TABLE_SUFFIXES = {".csv", ".parquet", ".tsv"}


def hash_participant_id(value) -> str:
    """Reproduce this pipeline's deidentified id: sub- + sha256(orig)[:8]."""
    return "sub-" + hashlib.sha256(str(value).encode()).hexdigest()[:8]


def _apply_id_hash(df: pd.DataFrame, cols) -> tuple:
    """Hash the given id columns (unique-mapped) so a reference frame's ids match ours."""
    hashed = []
    out = df
    for c in cols:
        if c in df.columns:
            if out is df:
                out = df.copy()
            uniq = out[c].astype(str).unique()
            mapping = {u: hash_participant_id(u) for u in uniq}
            out[c] = out[c].astype(str).map(mapping)
            hashed.append(c)
    return out, hashed


def load_table(path: Path) -> pd.DataFrame:
    """Load a tabular derivative (csv / csv.gz / tsv / parquet) into a DataFrame."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.name.endswith(".tsv") or path.name.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _ordered_diff(a, b):
    """(only in a, only in b) preserving order of each source."""
    sb, sa = set(b), set(a)
    return [c for c in a if c not in sb], [c for c in b if c not in sa]


def _sort_key_columns(common, df):
    """Non-float columns first (identifiers/labels), then the rest — a stable sort key."""
    labels = [c for c in common if not is_numeric_dtype(df[c]) or df[c].dtype == "int64"]
    rest = [c for c in common if c not in labels]
    return labels + rest


def _row_overlap(a: pd.DataFrame, b: pd.DataFrame, common) -> float:
    """Fraction of rows (over common columns) shared as a multiset, NaN- and float-safe."""

    def counts(df):
        d = df[common].copy()
        num = [c for c in common if is_numeric_dtype(d[c]) and not is_bool_dtype(d[c])]
        if num:
            d[num] = d[num].round(6)
        d = d.astype(str)  # uniform dtype; category/bool/NaN -> consistent strings
        key = d[common[0]]
        for c in common[1:]:
            key = key.str.cat(d[c], sep="\x1f")
        return key.value_counts()

    va, vb = counts(a), counts(b)
    idx = va.index.intersection(vb.index)
    shared = int(np.minimum(va.reindex(idx), vb.reindex(idx)).sum()) if len(idx) else 0
    denom = max(len(a), len(b)) or 1
    return shared / denom


def _sorted(df: pd.DataFrame, common, key) -> pd.DataFrame:
    """Order rows by the key columns cast to str, so both sides sort identically
    regardless of dtype (e.g. category ordering vs lexicographic)."""
    tmp = df[common].reset_index(drop=True)
    if not key:
        return tmp
    order = tmp[key].astype(str)
    return tmp.iloc[order.sort_values(list(key), kind="mergesort").index].reset_index(drop=True)


def compare_tables(a: pd.DataFrame, b: pd.DataFrame, rtol=RTOL, atol=ATOL,
                   hash_id_cols=HASH_ID_COLS) -> dict:
    """Structural + cell-level consistency between two tables (order-independent).

    Id columns in the reference frame ``a`` are hashed to this pipeline's namespace so
    rows align despite deidentification (see hash_participant_id).
    """
    a, hashed_id_cols = _apply_id_hash(a, hash_id_cols)
    cols_only_orig, cols_only_new = _ordered_diff(list(a.columns), list(b.columns))
    common = [c for c in a.columns if c in set(b.columns)]
    out = {
        "cols_only_orig": cols_only_orig,
        "cols_only_new": cols_only_new,
        "n_common_cols": len(common),
        "n_rows_orig": len(a),
        "n_rows_new": len(b),
        "hashed_id_cols": hashed_id_cols,
        "aligned": False,
        "match_fraction": None,
        "n_mismatch": None,
        "max_abs_diff": None,
        "per_column_mismatch": {},
        "row_overlap_fraction": None,
    }
    if not common:
        return out
    if len(a) != len(b):
        out["row_overlap_fraction"] = _row_overlap(a, b, common)
        return out

    # equal row counts -> align by sorting on shared label/key columns, then compare
    key = _sort_key_columns(common, a)
    a2 = _sorted(a, common, key)
    b2 = _sorted(b, common, key)

    n_mismatch = 0
    max_abs_diff = 0.0
    per_col = {}
    for c in common:
        av, bv = a2[c], b2[c]
        if is_numeric_dtype(av) and is_numeric_dtype(bv):
            an = av.to_numpy(dtype=float)
            bn = bv.to_numpy(dtype=float)
            close = np.isclose(an, bn, rtol=rtol, atol=atol, equal_nan=True)
            mism = int((~close).sum())
            diff = np.abs(an - bn)
            diff = diff[np.isfinite(diff)]
            if diff.size:
                max_abs_diff = max(max_abs_diff, float(diff.max()))
        else:
            eq = (av.to_numpy() == bv.to_numpy()) | (av.isna().to_numpy() & bv.isna().to_numpy())
            mism = int((~eq).sum())
        if mism:
            per_col[c] = mism
        n_mismatch += mism

    n_cells = len(a2) * len(common)
    out.update(
        aligned=True,
        n_mismatch=n_mismatch,
        max_abs_diff=max_abs_diff,
        match_fraction=(1.0 - n_mismatch / n_cells) if n_cells else 1.0,
        per_column_mismatch=per_col,
    )
    return out


def _walk(obj, prefix=""):
    """Yield (path, leaf_value) for a nested JSON-like structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare_json(a, b, rtol=RTOL, atol=ATOL) -> dict:
    """Leaf-by-leaf consistency of two JSON-like objects (numeric within tolerance)."""
    la, lb = dict(_walk(a)), dict(_walk(b))
    paths = set(la) | set(lb)
    mismatched = []
    for p in sorted(paths):
        if p not in la or p not in lb:
            mismatched.append(p)
            continue
        x, y = la[p], lb[p]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if not np.isclose(float(x), float(y), rtol=rtol, atol=atol):
                mismatched.append(p)
        elif x != y:
            mismatched.append(p)
    n = len(paths)
    return {
        "n_leaves": n,
        "n_mismatch": len(mismatched),
        "match_fraction": (1.0 - len(mismatched) / n) if n else 1.0,
        "mismatched_paths": mismatched,
    }


def _table_status(r: dict) -> str:
    if r["cols_only_orig"] or r["cols_only_new"] or r["n_rows_orig"] != r["n_rows_new"]:
        return "shape_mismatch"
    if r["n_mismatch"] == 0:
        return "identical" if r["max_abs_diff"] == 0 else "consistent"
    return "differs"


def compare_files(orig_path: Path, new_path: Path, rtol=RTOL, atol=ATOL,
                  hash_id_cols=HASH_ID_COLS) -> dict:
    """Compare one reference file to its pipeline counterpart; add kind and status."""
    orig_path, new_path = Path(orig_path), Path(new_path)
    suffix = orig_path.suffix
    base = {"file": orig_path.name}
    if suffix in _TABLE_SUFFIXES or orig_path.name.endswith(".tsv.gz"):
        r = compare_tables(load_table(orig_path), load_table(new_path), rtol, atol, hash_id_cols)
        return {**base, "kind": "table", "status": _table_status(r), **r}
    if suffix == ".json":
        r = compare_json(
            json.loads(orig_path.read_text()), json.loads(new_path.read_text()), rtol, atol
        )
        status = "identical" if r["n_mismatch"] == 0 else "differs"
        return {**base, "kind": "json", "status": status, **r}
    identical = orig_path.read_bytes() == new_path.read_bytes()
    return {
        "file": orig_path.name,
        "kind": "binary",
        "status": "identical" if identical else "differs",
        "note": "binary; byte comparison only",
    }


def compare_directories(orig_dir: Path, new_dir: Path, rtol=RTOL, atol=ATOL,
                        hash_id_cols=HASH_ID_COLS) -> list:
    """Compare every file in ``orig_dir`` to its counterpart in ``new_dir``."""
    orig_dir, new_dir = Path(orig_dir), Path(new_dir)
    results = []
    for p in sorted(f for f in orig_dir.glob("*") if f.is_file()):
        counterpart = new_dir / p.name
        if not counterpart.exists():
            results.append({"file": p.name, "kind": "-", "status": "missing"})
        else:
            results.append(compare_files(p, counterpart, rtol, atol, hash_id_cols))
    return results


_REPORT_COLS = [
    "file", "kind", "status", "n_rows_orig", "n_rows_new",
    "cols_only_orig", "cols_only_new", "hashed_id_cols", "match_fraction",
    "max_abs_diff", "n_mismatch", "row_overlap_fraction",
]


def report_dataframe(results: list) -> pd.DataFrame:
    """Flatten comparison results into a tabular report."""
    rows = []
    for r in results:
        row = {c: r.get(c) for c in _REPORT_COLS}
        for c in ("cols_only_orig", "cols_only_new", "hashed_id_cols"):
            if isinstance(row[c], list):
                row[c] = ";".join(row[c])
        rows.append(row)
    return pd.DataFrame(rows, columns=_REPORT_COLS)


# Statuses that mean a derivative does not match the reference: value
# differences, structural (shape/column) differences, or an absent counterpart.
# ``consistent`` (equal within numeric tolerance) and ``identical`` are passes.
INCONSISTENT_STATUSES = ("differs", "shape_mismatch", "missing")


def inconsistent_results(results: list) -> list:
    """Return the result rows whose status signals a mismatch with the reference."""
    return [r for r in results if r.get("status") in INCONSISTENT_STATUSES]


def format_failure_banner(failing: list) -> str:
    """A prominent, hard-to-miss banner listing files that do not match the reference."""
    bar = "!" * 80
    lines = [
        bar,
        "VALIDATION FAILED",
        "Workflow results do not match the intended (reference) results.",
        f"{len(failing)} file(s) inconsistent with the benchmark data:",
    ]
    for r in failing:
        lines.append(f"    - {r['file']}  ({r['status']})")
    lines.append(bar)
    return "\n".join(lines)


def format_success_banner(results: list) -> str:
    """A prominent banner confirming every file matched the benchmark reference."""
    bar = "=" * 80
    return "\n".join(
        [
            bar,
            f"SUCCESS: All results match benchmarks ({len(results)} files).",
            bar,
        ]
    )


def format_report(results: list) -> str:
    """Human-readable per-file summary."""
    df = report_dataframe(results)
    lines = ["Derivatives consistency vs reference:", ""]
    lines.append(df.to_string(index=False))
    counts = df["status"].value_counts().to_dict()
    lines.append("")
    lines.append("Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)
