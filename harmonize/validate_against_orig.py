#!/usr/bin/env python3
"""Compare pipeline derivatives to a reference set and report per-file consistency.

By default compares config.DERIV against the in-repo ``benchmark_data`` reference
set, printing a summary and writing validation_report.csv.
"""

import argparse
import sys

from abcd_ksads import config
from abcd_ksads.validate import (
    compare_directories,
    format_failure_banner,
    format_report,
    inconsistent_results,
    report_dataframe,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orig", default=config.BENCHMARK)
    ap.add_argument("--new", default=config.DERIV)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-8)
    args = ap.parse_args()

    results = compare_directories(args.orig, args.new, rtol=args.rtol, atol=args.atol)
    print(format_report(results))

    out = config.DERIV / "validation_report.csv"
    report_dataframe(results).to_csv(out, index=False)
    print(f"\nWrote {out.as_posix()}")

    # Fail loudly (and non-zero, so `make` stops) if any derivative diverges.
    failing = inconsistent_results(results)
    if failing:
        sys.stdout.flush()  # keep the banner last, after the report
        print("\n" + format_failure_banner(failing), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
