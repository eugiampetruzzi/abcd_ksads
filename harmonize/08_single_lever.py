#!/usr/bin/env python3
import pandas as pd
from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_crosswalk
from abcd_ksads.multiverse import (
    build_primitive_cache,
    prevalence,
    construct_status,
    BASE_SES,
)


def main():
    cw = build_crosswalk()
    resolved = pd.read_parquet(config.DERIV / "ksads_resolved_long.parquet")
    for c in ["session_id", "variable", "resolved"]:
        resolved[c] = resolved[c].astype(str)
    base = resolved[resolved.session_id == BASE_SES].copy()
    cache = build_primitive_cache(base, cw)

    def prev(status, informant, subthr, phobia):
        stat = construct_status(
            cache, "any-disorder", status, informant, subthr, phobia
        )
        return prevalence(stat)[0]

    base_cfg = dict(
        status="current", informant="parent", subthr=False, phobia="phobia_in"
    )
    base_prev = prev(**base_cfg)

    flips = [
        ("current -> ever-met", dict(base_cfg, status="ever_met")),
        ("parent -> youth-only", dict(base_cfg, informant="youth")),
        ("parent -> either", dict(base_cfg, informant="either")),
        ("+ subthreshold dx", dict(base_cfg, subthr=True)),
        ("anxiety: drop phobia", dict(base_cfg, phobia="phobia_out")),
    ]
    rows = [
        {
            "lever": "base (current, parent, full, phobia-in)",
            "prevalence_pct": round(base_prev, 3),
            "delta_pts": 0.0,
        }
    ]
    for name, cfg in flips:
        p = prev(**cfg)
        rows.append(
            {
                "lever": name,
                "prevalence_pct": round(p, 3),
                "delta_pts": round(p - base_prev, 3),
            }
        )
    df = pd.DataFrame(rows)
    body = df.iloc[1:].reindex(
        df.iloc[1:].delta_pts.abs().sort_values(ascending=False).index
    )
    df = pd.concat([df.iloc[[0]], body], ignore_index=True)
    df.to_csv(config.DERIV / "single_lever.csv", index=False)

    print(df.to_string(index=False))
    print(f"\nWrote {config.DERIV.as_posix()}/single_lever.csv")


if __name__ == "__main__":
    main()
