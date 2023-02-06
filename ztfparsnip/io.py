#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import os, re, logging, yaml

from typing import List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

if os.getenv("ZTFDATA"):
    BTS_LC_BASELINE_DIR = os.path.join(
        str(os.getenv("ZTFDATA")), "nuclear_sample", "BTS", "baseline"
    )
    TRAIN_DATA = os.path.join(
        str(os.getenv("ZTFDATA")), "nuclear_sample", "BTS", "train"
    )
    for d in [BTS_LC_BASELINE_DIR, TRAIN_DATA]:
        if not os.path.exists(d):
            os.makedirs(d)

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


def add_mag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mag and magerr
    """
    # remove negative flux rows
    df.query("ampl_corr>0", inplace=True)

    F0 = 10 ** (df.magzp / 2.5)
    F0_err = F0 / 2.5 * np.log(10) * df.magzpunc
    flux = df.ampl_corr / F0 * 3630.78
    flux_err = (
        np.sqrt((df.ampl_err_corr / F0) ** 2 + (df.ampl_corr * F0_err / F0**2) ** 2)
        * 3630.78
    )

    # convert to magsubl
    abmag = -2.5 * np.log10(flux / 3630.78)
    abmag_err = 2.5 / np.log(10) * flux_err / flux

    df["magpsf"] = abmag
    df["sigmapsf"] = abmag_err

    return df


def get_lightcurve(
    ztfid: str, lc_dir: str | None = None, enforce_z: bool = True
) -> (None | pd.DataFrame, None | dict):
    if is_valid_ztfid(ztfid):
        if lc_dir is None:
            lc_dir = BTS_LC_BASELINE_DIR
    lc = get_ztfid_dataframe(ztfid=ztfid, lc_dir=lc_dir)
    header = get_ztfid_header(ztfid=ztfid, lc_dir=lc_dir)

    if enforce_z and header.get("bts_z") == "-":
        return None, None

    return lc, header


def get_ztfid_dataframe(ztfid: str, lc_dir: str | None = None) -> pd.DataFrame | None:
    """
    Get the Pandas Dataframe of a single transient
    """
    if is_valid_ztfid(ztfid):
        if lc_dir is None:
            lc_dir = BTS_LC_BASELINE_DIR
        filepath = os.path.join(lc_dir, f"{ztfid}_bl.csv")

        try:
            df = pd.read_csv(filepath, comment="#", index_col=0)
            df["jd"] = df.obsmjd + 2400000.5
            df_with_mag = add_mag(df)
            return df_with_mag
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


def get_all_ztfids(lc_dir: str | None = None) -> List[str]:
    """
    Checks the lightcurve folder and gets all ztfids
    """
    if lc_dir is None:
        lc_dir = BTS_LC_BASELINE_DIR

    ztfids = []
    for name in os.listdir(lc_dir):
        if name[-4:] == ".csv":
            if "_bl" in name[:-4]:
                ztfid = name[:-7]
            else:
                ztfid = name[:-4]
            ztfids.append(ztfid)
    return ztfids


def load_config(config_path: str | None = None) -> dict:
    """
    Loads the user-specific config
    """
    if not config_path:
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "..", "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config
