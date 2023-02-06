#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os

if os.getenv("ZTFDATA"):
    BTS_LC_DIR = os.path.join(
        str(os.getenv("ZTFDATA")), "nuclear_sample", "BTS", "data"
    )
    BTS_LC_BASELINE_DIR = os.path.join(
        str(os.getenv("ZTFDATA")), "nuclear_sample", "BTS", "baseline"
    )

    for path in [BTS_LC_DIR, BTS_LC_BASELINE_DIR]:
        if not os.path.exists(path):
            os.makedirs(path)

else:
    raise ValueError(
        "You have to set the ZTFDATA environment variable in your .bashrc or .zshrc. See github.com/mickaelrigault/ztfquery"
    )
