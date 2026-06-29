#!/usr/bin/env python3
import importlib.util
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DERIV = os.path.join(HERE, "derivatives")


def _load(f):
    spec = importlib.util.spec_from_file_location(f[:-3].replace(".", "_"),
                                                  os.path.join(HERE, f))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


MV = _load("06_multiverse_spec.py")


def main():
    cw = MV.L3.build_crosswalk()
    resolved = pd.read_parquet(os.path.join(DERIV, "ksads_resolved_long.parquet"))
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)
    base = resolved[resolved.session_id == MV.BASE_SES].copy()
    cache = MV.build_primitive_cache(base, cw)

    def prev(status, informant, subthr, phobia):
        stat = MV.construct_status(cache, "any-disorder", status, informant, subthr, phobia)
        return MV.prevalence(stat)[0]

    base_cfg = dict(status="current", informant="parent", subthr=False, phobia="phobia_in")
    base_prev = prev(**base_cfg)

    flips = [
        ("current -> ever-met",      dict(base_cfg, status="ever_met")),
        ("parent -> youth-only",     dict(base_cfg, informant="youth")),
        ("parent -> either",         dict(base_cfg, informant="either")),
        ("+ subthreshold dx",        dict(base_cfg, subthr=True)),
        ("anxiety: drop phobia",     dict(base_cfg, phobia="phobia_out")),
    ]
    rows = [{"lever": "base (current, parent, full, phobia-in)",
             "prevalence_pct": round(base_prev, 3), "delta_pts": 0.0}]
    for name, cfg in flips:
        p = prev(**cfg)
        rows.append({"lever": name, "prevalence_pct": round(p, 3),
                     "delta_pts": round(p - base_prev, 3)})
    df = pd.DataFrame(rows)
    body = df.iloc[1:].reindex(df.iloc[1:].delta_pts.abs().sort_values(ascending=False).index)
    df = pd.concat([df.iloc[[0]], body], ignore_index=True)
    df.to_csv(os.path.join(DERIV, "single_lever.csv"), index=False)

    print(df.to_string(index=False))
    print(f"\nWrote {DERIV}/single_lever.csv")


if __name__ == "__main__":
    main()