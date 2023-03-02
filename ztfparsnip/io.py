#!/usr/bin/env python3
# Author: Simeon Reusch (simeon.reusch@desy.de)
# License: BSD-3-Clause

import logging
import os
import random
import re
import string
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from astropy.time import Time  # type: ignore

logger = logging.getLogger(__name__)
alphabet = string.ascii_lowercase + string.digits

if ztfdir := os.getenv("ZTFDATA"):
    BASE_DIR = Path(ztfdir) / "ztfparsnip"
    BTS_LC_BASELINE_DIR = BASE_DIR / "BTS_plus_TDE"
    TRAIN_DATA = BASE_DIR / "train"
    PLOT_DIR = BASE_DIR / "plots"
    DOWNLOAD_URL_SAMPLE = Path(
        "https://syncandshare.desy.de/index.php/s/cQHcnXmYDzRGyqG/download"
    )
    DOWNLOAD_URL_SAMPLE_TEST = Path(
        "https://syncandshare.desy.de/index.php/s/bnGQYb9goiHi6bH/download"
    )

    for d in [BASE_DIR, BTS_LC_BASELINE_DIR, TRAIN_DATA, PLOT_DIR]:
        if not d.is_dir():
            os.makedirs(d)

    BTS_HEADERS = BTS_LC_BASELINE_DIR / "headers.json"

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


def get_lightcurve(ztfid: str, lc_dir: Path | None = None):
    if is_valid_ztfid(ztfid):
        if lc_dir is None:
            lc_dir = BTS_LC_BASELINE_DIR
    lc = get_ztfid_dataframe(ztfid=ztfid, lc_dir=lc_dir)
    header = get_ztfid_header(ztfid=ztfid, lc_dir=lc_dir)

    config = load_config()

    if header is not None:
        if header.get("bts_class") in config["simpleclasses"]["star"]:
            header["bts_z"] = 0

    if header is not None:
        if header.get("bts_z") in ["-", None]:
            return None, header

    return lc, header


def get_ztfid_dataframe(ztfid: str, lc_dir: Path | None = None) -> pd.DataFrame | None:
    """
    Get the Pandas Dataframe of a single transient
    """
    if is_valid_ztfid(ztfid):
        if lc_dir is None:
            lc_dir = BTS_LC_BASELINE_DIR
        filepath = lc_dir / f"{ztfid}_bl.csv"

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


def get_ztfid_header(ztfid: str, lc_dir: Path | None = None) -> dict | None:
    """
    Returns the metadata contained in the csvs as dictionary
    """
    if is_valid_ztfid(ztfid):
        if lc_dir is None:
            lc_dir = BTS_LC_BASELINE_DIR
        filepath = lc_dir / f"{ztfid}_bl.csv"

        try:
            with open(filepath, "r") as input_file:
                headerkeys = []
                headervals = []

                for i, line in enumerate(input_file):
                    if len(line) >= 300:
                        break
                    if line == "\n":
                        break
                    if ",ampl_corr" in line:
                        break
                    key = line.split(",", 2)[0].split("=")[0].lstrip("#")
                    headerkeys.append(key)
                    val = line.split(",", 2)[0].split("=")[1][:-1]
                    headervals.append(val)

                returndict = {}
                for i, key in enumerate(headerkeys):
                    if headervals[i] in ["-", None, "None"]:
                        returnval = "-"
                    else:
                        returnval = headervals[i]
                    returndict.update({key: returnval})

                returndict["ztfid"] = returndict["name"]

                return returndict

        except FileNotFoundError:
            logger.warn(f"No file found for {ztfid}. Check the ID.")
            return None
    else:
        raise ValueError(f"{ztfid} is not a valid ZTF ID")


def save_csv_with_header(lc, savedir: Path, output_format: str = "ztfnuclear"):
    """
    Generate a string of the header from a dict, meant to be written to a csv file. Save the lightcurve with the header info as csv
    """
    header = lc.meta

    lc_id = lc.meta.get("name")

    if "_" in lc_id:
        parent_ztfid = lc_id.split("_")[0]
    else:
        parent_ztfid = lc_id

    header["parent_ztfid"] = parent_ztfid

    if output_format == "ztfnuclear":
        band_to_filterid = {"ztfg": 1, "ztfr": 2, "ztfi": 3}
        del lc["zpsys"]

        lc["obsmjd"] = lc["jd"] - 2400000.5
        del lc["jd"]

        lc.rename_column("zp", "magzp")
        lc.rename_column("flux", "ampl_corr")
        lc.rename_column("fluxerr", "ampl_err_corr")

    filename = f"{lc_id}.csv"

    headerstr = ""
    for i, val in header.items():
        headerstr += f"#{i}={val}\n"

    outfile = savedir / filename

    df = lc.to_pandas()
    # Time index for use for rolling window
    df = df.sort_values("obsmjd")
    obs_jd = Time(df["obsmjd"].values, format="mjd")
    df = df.set_index(pd.to_datetime(obs_jd.datetime))

    if os.path.isfile(outfile):
        os.remove(outfile)

    with open(outfile, "w") as f:
        f.write(headerstr)
        df.to_csv(f)
        f.close()


def short_id():
    return "".join(random.choices(alphabet, k=5))


def get_all_ztfids(lc_dir: Path | None = None, test: bool = False) -> List[str]:
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

    if test:
        config = load_config()
        ztfids = [ztfid for ztfid in ztfids if ztfid in config["test_lightcurves"]]

    return ztfids


def load_config(config_path: Path | None = None) -> dict:
    """
    Loads the user-specific config
    """
    if not config_path:
        current_dir = Path(__file__).parent
        config_path = current_dir.parent / "config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def download_sample(test: bool = False):
    """
    Download the BTS + TDE lightcurves from the DESY Nextcloud
    """
    if ZTFDATA := os.getenv("ZTFDATA"):
        if test:
            cmd_dl = f"curl --create-dirs -J -O --output-dir {ZTFDATA}/ztfparsnip {DOWNLOAD_URL_SAMPLE_TEST}"
        else:
            cmd_dl = f"curl --create-dirs -J -O --output-dir {ZTFDATA}/ztfparsnip {DOWNLOAD_URL_SAMPLE}"
        cmd_extract = (
            f"unzip {ZTFDATA}/ztfparsnip/BTS_plus_TDE.zip -d {ZTFDATA}/ztfparsnip"
        )
        cmd_remove_zip = f"rm {ZTFDATA}/ztfparsnip/BTS_plus_TDE.zip"

        # Download
        subprocess.run(cmd_dl, shell=True)
        logger.info(f"Sample download complete, extracting files")

        # Extract
        subprocess.run(cmd_extract, shell=True)
        extracted_dir = Path(ZTFDATA) / "ztfparsnip" / "BTS_plus_TDE"

        # Validate
        nr_files = len([x for x in extracted_dir.glob("*") if x.is_file()])
        if nr_files == 6841 and test:
            subprocess.run(cmd_remove_zip, shell=True)
        elif nr_files == 10 and test:
            subprocess.run(cmd_remove_zip, shell=True)
        else:
            raise ValueError(
                "Something went wrong with your download. Remove 'ZTFDATA/ztfparsnip/BTS_plus_TDE' and try again"
            )

    else:
        raise ValueError(
            "You have to set the ZTFDATA environment variable in your .bashrc or .zshrc. See https://github.com/mickaelrigault/ztfquery"
        )
