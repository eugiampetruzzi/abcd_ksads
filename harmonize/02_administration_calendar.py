#!/usr/bin/env python3
"""Module x wave x informant administration calendar; the classification and cadence
logic lives in abcd_ksads.administration."""

import pandas as pd

from abcd_ksads import config
from abcd_ksads.administration import SESSIONS, V2_SWITCH, build_calendar


def main():
    s = pd.read_csv(config.DERIV / "ksads_resolution_summary.csv")
    long, grid = build_calendar(s)

    long.to_csv(config.DERIV / "ksads_administration_calendar.csv", index=False)
    grid.to_csv(config.DERIV / "ksads_administration_grid.csv", index=False)

    # report
    print("Layer 2 administration calendar")
    print("  X = administered, . = not administered (all 555), blank = absent")
    print(f"  V2 switch at {V2_SWITCH}\n")
    cur = None
    for _, r in grid.iterrows():
        if r["informant"] != cur:
            cur = r["informant"]
            print(f"\n[{cur.upper()}]  " + " ".join(w[-3:] for w in SESSIONS))
        cells = " ".join(f"{str(r[w]):>3}" for w in SESSIONS)
        fl = f"  <-- {r['flag']}" if r["flag"] else ""
        print(f"  {r['module']:8} {cells}{fl}")
    nadd = grid.flag.str.contains("added@").sum()
    ndrop = grid.flag.str.contains("dropped_after@").sum()
    print(f"\nFlagged: {nadd} added mid-study, {ndrop} dropped before the final wave.")
    print(f"\nWrote {config.DERIV.as_posix()}/ksads_administration_calendar.csv")
    print(f"Wrote {config.DERIV.as_posix()}/ksads_administration_grid.csv")


if __name__ == "__main__":
    main()
