#!/usr/bin/env python3
"""Ingest the raw ABCD phenotype TSVs into the consolidated Parquet cache.

Before ingesting, verifies that every file listed in config.PHENOTYPE_MANIFEST is
present in config.RAW_PHENOTYPE; if any are missing it prints a readable report and
exits without writing anything. Otherwise it consolidates every TSV into one wide
phenotype.parquet and writes metadata.json from the JSON sidecars.
"""

import sys

from abcd_ksads import config
from abcd_ksads.ingest import MissingFilesError, ingest


def main():
    try:
        summary = ingest(
            config.RAW_PHENOTYPE,
            config.RAW_CACHE,
            manifest_path=config.PHENOTYPE_MANIFEST,
        )
    except MissingFilesError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    print(
        f"Consolidated {summary['n_tables']} tables into "
        f"{summary['n_rows']:,} rows x {summary['n_columns']:,} variables"
    )
    print(f"Wrote {config.RAW_CACHE.as_posix()}/phenotype.parquet")
    print(f"Wrote {config.RAW_CACHE.as_posix()}/metadata.json")


if __name__ == "__main__":
    main()
