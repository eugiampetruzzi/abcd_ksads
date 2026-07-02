#!/usr/bin/env python3
from abcd_ksads import config
from abcd_ksads.category_crosswalk import build_crosswalk


if __name__ == "__main__":
    cw = build_crosswalk()
    cw.to_csv(config.DERIV / "ksads_category_crosswalk.csv", index=False)
