#!/usr/bin/env python3
"""Ingest the raw ABCD phenotype TSVs into the Parquet cache (config.RAW_CACHE).

Reads every TSV in config.RAW_PHENOTYPE, mirrors each to Parquet with values
preserved verbatim, and writes a consolidated metadata.json from the JSON sidecars.
"""

from abcd_ksads import config
from abcd_ksads.ingest import ingest


def main():
    summary = ingest(config.RAW_PHENOTYPE, config.RAW_CACHE)
    print(
        f"Consolidated {summary['n_tables']} tables into "
        f"{summary['n_rows']:,} rows x {summary['n_columns']:,} variables"
    )
    print(f"Wrote {config.RAW_CACHE.as_posix()}/phenotype.parquet")
    print(f"Wrote {config.RAW_CACHE.as_posix()}/metadata.json")


if __name__ == "__main__":
    main()
