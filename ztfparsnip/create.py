#!/usr/bin/env python3
# Author: Lightcurve creation code by Alice Townsend (alice.townsend@hu-berlin.de)
# License: BSD-3-Clause

import os, numpy, logging, json

from typing import Any

from tqdm import tqdm

from ztfparsnip import io
from ztfparsnip import noisify
import lcdata


class CreateLightcurves(object):
    """
    This is the parent class for the ZTF nuclear transient sample"""

    def __init__(
        self,
        weights: dict[Any] = {},
        bts_baseline_dir: str | None = None,
        name: str = "train",
        reprocess_headers: bool = False,
    ):
        super(CreateLightcurves, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Creating lightcurves")
        self.weights = weights
        self.name = name

        if bts_baseline_dir is None:
            self.lc_dir = io.BTS_LC_BASELINE_DIR
        else:
            self.lc_dir = bts_baseline_dir

        self.ztfids = io.get_all_ztfids(lc_dir=self.lc_dir)
        self.config = io.load_config()

        self.get_headers(reprocess=reprocess_headers)

    def get_simple_class(self, bts_class: str) -> str:
        """
        Look in the config file to get the simple classification for a transient
        """
        for key, val in self.config["simpleclasses"].items():
            for entry in self.config["simpleclasses"][key]:
                if bts_class == entry or bts_class == f"SN {entry}":
                    return key
        return "unclass"

    def get_lightcurves(self, start: int = 0, end: int | None = None):
        """
        Read dataframes and headers
        """
        if end is None:
            end = len(self.ztfids)

        for ztfid in tqdm(self.ztfids[start:end], total=len(self.ztfids[start:end])):
            lc, header = io.get_lightcurve(ztfid=ztfid, lc_dir=self.lc_dir)
            if lc is not None:
                bts_class = header.get("bts_class")
                simple_class = self.get_simple_class(bts_class)
                header["simple_class"] = simple_class
                yield lc, header
            else:
                yield None, header

    def get_headers(self, reprocess: bool = False):
        """
        Read headers only
        """
        headers_raw = {}
        self.headers = {}

        if not os.path.isfile(io.BTS_HEADERS) or reprocess:

            for ztfid in tqdm(self.ztfids, total=len(self.ztfids)):
                _, header = io.get_lightcurve(ztfid=ztfid, lc_dir=self.lc_dir)
                header["simple_class"] = self.get_simple_class(header.get("bts_class"))
                headers_raw.update({header.get("name"): header})

            with open(io.BTS_HEADERS, "w") as f:
                f.write(json.dumps(headers_raw))

        else:
            with open(io.BTS_HEADERS, "r") as f:
                headers_raw = json.load(f)

        for k, v in headers_raw.items():
            if (z := v.get("bts_z")) != "-" and z is not None:
                self.headers.update({k: v})

    def select(self, seed=None):
        """
        Select initial lightcurves based on weights and classifications
        """
        class_stats = {}

        if len(self.weights) == 0:
            raise ValueError(
                "You have to define weights. Pass the weights dictionary to the CreateLightcurves class"
            )

        # Check if we do relative amounts of lightcurves or absolute
        weight_values = list(self.weights.values())
        if isinstance(weight_values[0], float):
            for val in weight_values:
                assert isinstance(val, float)
            relative_weighting = True
        else:
            for val in weight_values:
                assert isinstance(val, int)
            relative_weighting = False

        # Now we count classes
        for c in self.config["simpleclasses"]:
            for entry in self.headers.values():
                print(entry.get("simple_class"))

    def noisify(self, train_dir: str = None):
        """
        Noisify the sample
        """
        failed = {"no_z": [], "no_class": [], "no_lc_after_cuts": []}

        bts_lc_list = []
        noisy_lc_list = []
        for lc, header in self.get_lightcurves(1):
            if lc is not None:
                if header.get("bts_class") is not None:
                    bts_lc, noisy_lc = noisify.noisify_lightcurve(lc, header)
                    if bts_lc is not None:
                        bts_lc_list.append(bts_lc)
                        noisy_lc_list.extend(noisy_lc)
                    else:
                        failed["no_lc_after_cuts"].append(header.get("name"))
                else:
                    failed["no_class"].append(header.get("name"))
            else:
                failed["no_z"].append(header.get("name"))

        lc_list = [*bts_lc_list, *noisy_lc_list]

        if train_dir is None:
            train_dir = io.TRAIN_DATA
        else:
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)

        self.logger.info(
            f"{len(failed['no_z'])} items: no redshift | {len(failed['no_lc_after_cuts'])} items: lc does not survive cuts | {len(failed['no_class'])} items: no classification"
        )

        self.logger.info(
            f"Generated {len(noisy_lc_list)} noisy lightcurves from {len(bts_lc_list)} original lightcurves"
        )

        # Save h5 files
        dataset_h5_bts = lcdata.from_light_curves(bts_lc_list)
        dataset_h5_noisy = lcdata.from_light_curves(noisy_lc_list)
        dataset_h5_combined = lcdata.from_light_curves(lc_list)

        dataset_h5_bts.write_hdf5(
            os.path.join(train_dir, f"{self.name}_bts.h5"), overwrite=True
        )
        dataset_h5_noisy.write_hdf5(
            os.path.join(train_dir, f"{self.name}_noisy.h5"), overwrite=True
        )
        dataset_h5_combined.write_hdf5(
            os.path.join(train_dir, f"{self.name}_combined.h5"), overwrite=True
        )
