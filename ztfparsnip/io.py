#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, re

import pandas as pd

if os.getenv("ZTFDATA"):
    BTS_LC_BASELINE_DIR = os.path.join(
        str(os.getenv("ZTFDATA")), "nuclear_sample", "BTS", "baseline"
    )
    if not os.path.exists(BTS_LC_BASELINE_DIR):
        os.makedirs(path)

else:
    raise ValueError(
        "You have to set the ZTFDATA environment variable in your .bashrc or .zshrc. See github.com/mickaelrigault/ztfquery"
    )


def is_valid_ztfid(ztfid: str) -> bool:
    """
    Checks if a string adheres to the ZTF naming scheme
    """
    is_match = re.match("^ZTF[1-2]\d[a-z]{7}$", ztfid)
    if is_match:
        return True
    else:
        return False


def get_ztfid_dataframe(ztfid: str, lc_dir: str | None = None) -> pd.DataFrame | None:
    """
    Get the Pandas Dataframe of a single transient
    """
    if is_valid_ztfid(ztfid):
        if lc_dir is None:
            lc_dir = BTS_LC_BASELINE_DIR

        filepath = os.path.join(lc_dir, f"{ztfid}_bl.csv")

        try:
            df = pd.read_csv(filepath, comment="#")
            return df
        except FileNotFoundError:
            logger.warn(f"No file found for {ztfid}. Check the ID.")
            return None
    else:
        raise ValueError(f"{ztfid} is not a valid ZTF ID")


def get_ztfid_header(ztfid: str, lc_dir: str | None = None) -> dict | None:
    """
    Returns the metadata contained in the csvs as dictionary
    """
    if is_valid_ztfid(ztfid):
        if lc_dir is None:
            lc_dir = BTS_LC_BASELINE_DIR

        filepath = os.path.join(lc_dir, f"{ztfid}_bl.csv")

        try:
            with open(filepath, "r") as input_file:
                headerkeys = []
                headervals = []

                for i, line in enumerate(input_file):
                    if len(line) >= 300:
                        break
                    key = line.split(",", 2)[0].split("=")[0].lstrip("#")
                    headerkeys.append(key)
                    val = line.split(",", 2)[0].split("=")[1][:-1]
                    headervals.append(val)

                returndict = {}
                for i, key in enumerate(headerkeys):
                    returndict.update({key: headervals[i]})

                returndict["ztfid"] = returndict.get("name")

                return returndict

        except FileNotFoundError:
            logger.warn(f"No file found for {ztfid}. Check the ID.")
            return None
    else:
        raise ValueError(f"{ztfid} is not a valid ZTF ID")
