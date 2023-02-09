#!/usr/bin/env python3
# Author: Lightcurve creation code by Alice Townsend (alice.townsend@hu-berlin.de)
# License: BSD-3-Clause

import os, numpy, logging, json, math

from pathlib import Path

from typing import Any
from numpy.random import default_rng

from tqdm import tqdm

from ztfparsnip import io
from ztfparsnip import noisify
from ztfparsnip import plot
import lcdata

import matplotlib.pyplot as plt


class CreateLightcurves(object):
    """
    This is the parent class for the ZTF nuclear transient sample"""

    def __init__(
        self,
        weights: None | dict[Any] = None,
        validation_fraction: float = 0.1,
        validation_seed: int | None = None,
        bts_baseline_dir: str | None = None,
        name: str = "train",
        reprocess_headers: bool = False,
        output_format: str = "h5",
        train_dir: Path = io.TRAIN_DATA,
    ):
        super(CreateLightcurves, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Creating lightcurves")
        self.weights = weights
        self.validation_fraction = validation_fraction
        self.validation_seed = validation_seed
        self.name = name
        self.output_format = output_format
        self.train_dir = train_dir

        self.rng = default_rng(seed=validation_seed)

        if isinstance(self.train_dir, str):
            self.train_dir = Path(self.train_dir)

        if not self.train_dir.exists():
            os.makedirs(self.train_dir)

        if bts_baseline_dir is None:
            self.lc_dir = io.BTS_LC_BASELINE_DIR
        else:
            self.lc_dir = bts_baseline_dir

        self.ztfids = io.get_all_ztfids(lc_dir=self.lc_dir)
        self.config = io.load_config()

        self.get_headers(reprocess=reprocess_headers)

        # initialize default weights
        if self.weights is None:
            self.weights = {
                "sn_ia": 12520,
                "tde": 12520,
                "sn_other": 12520,
            }

        weights_info = "\n"
        for k, v in self.weights.items():
            weights_info += f"{k}: {v}\n"

        self.logger.info("Creating noisified training data.")
        self.logger.info(
            f"\n---------------------------------\nSelected configuration\nweights: {weights_info}\nvalidation fraction: {self.validation_fraction}\nvalidation seed: {self.validation_seed}\noutput format: {self.output_format}\ntraining data output directory: {self.train_dir}\n---------------------------------"
        )

    def get_simple_class(self, bts_class: str) -> str:
        """
        Look in the config file to get the simple classification for a transient
        """
        for key, val in self.config["simpleclasses"].items():
            for entry in self.config["simpleclasses"][key]:
                if bts_class == entry or bts_class == f"SN {entry}":
                    return key
        return "unclass"

    def get_lightcurves(
        self, start: int = 0, end: int | None = None, ztfids: list | None = None
    ):
        """
        Read dataframes and headers
        """
        if end is None:
            end = len(self.ztfids)

        if ztfids is None:
            ztfids = self.ztfids

        for ztfid in tqdm(ztfids[start:end], total=len(ztfids[start:end])):
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

        if not io.BTS_HEADERS.is_file() or reprocess:

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

    def select(
        self,
    ):
        """
        Select initial lightcurves based on weights and classifications
        """
        classes_available = {}
        self.selection = {}
        self.validation_sample = {"all": {"ztfids": [], "entries": 0}}

        # Check if we do relative amounts of lightcurves or absolute
        weight_values = list(self.weights.values())
        if isinstance(weight_values[0], float):
            for val in weight_values:
                assert isinstance(val, float)
            relative_weighting = True
            raise ValueError("Not implemented yet. Please pass integers.")
        else:
            for val in weight_values:
                assert isinstance(val, int)
            relative_weighting = False

        # Now we count classes
        for c in self.config["simpleclasses"]:
            classes_available.update({c: {"ztfids": []}})
            for entry in self.headers.values():
                if entry.get("simple_class") == c:
                    classes_available.get(c).get("ztfids").append(entry.get("name"))
            classes_available[c]["entries"] = len(
                classes_available.get(c).get("ztfids")
            )

        for c in self.weights:
            if c not in classes_available.keys():
                raise ValueError(
                    f"Your weight names have to be in {list(classes_available.keys())}"
                )

        if relative_weighting is True:
            raise ValueError("Relative weighting is not implemented yet")

        availability = ""
        for k, v in classes_available.items():
            availability += f"{k}: {classes_available[k]['entries']}\n"
        self.logger.info(
            f"\n---------------------------------\nLightcurves available:\n{availability}---------------------------------"
        )
        for k, v in classes_available.items():
            validation_number = math.ceil(
                self.validation_fraction * classes_available[k]["entries"]
            )
            validation_ztfids = self.rng.choice(
                classes_available[k].get("ztfids"), size=validation_number
            )
            all_validation_ztfids = self.validation_sample["all"]["ztfids"]
            all_validation_ztfids.extend(validation_ztfids)
            self.validation_sample.update(
                {
                    k: {"ztfids": validation_ztfids, "entries": len(validation_ztfids)},
                    "all": {
                        "entries": self.validation_sample["all"]["entries"]
                        + len(validation_ztfids),
                        "ztfids": all_validation_ztfids,
                    },
                }
            )

        if relative_weighting is False:
            for c, target_n in self.weights.items():
                available = classes_available.get(c).get("entries")
                multiplier = int(target_n / available)

                self.selection.update({c: multiplier})

        expected = {}
        total = 0
        for k, v in self.selection.items():
            exp = classes_available[k]["entries"] * v
            expected.update({k: exp})
            total += exp

        self.logger.info(f"Your selected weights: {self.selection}")

        self.logger.info(f"Expected lightcurves: {expected} ({total} in total)")

        self.classes_available = classes_available

    def noisify(self, plot_debug: bool = False, n: int | None = None):
        """
        Noisify the sample
        """
        failed = {"no_z": [], "no_class": [], "no_lc_after_cuts": []}

        validation_lc_list = []
        bts_lc_list = []
        noisy_lc_list = []
        generated = {k: 0 for (k, v) in self.selection.items()}

        for lc, header in self.get_lightcurves(end=n):
            if lc is not None:
                if (c := header.get("simple_class")) is not None:
                    if c in self.selection.keys():

                        # check if it's a validation sample lightcurve
                        if header["name"] in self.validation_sample["all"]["ztfids"]:
                            validation_lc, _ = noisify.noisify_lightcurve(
                                table=lc, header=header, multiplier=0
                            )
                            if validation_lc is not None:
                                validation_lc_list.append(validation_lc)

                        else:
                            bts_lc, noisy_lcs = noisify.noisify_lightcurve(
                                table=lc, header=header, multiplier=self.selection[c]
                            )
                            if bts_lc is not None:
                                bts_lc_list.append(bts_lc)
                                noisy_lc_list.extend(noisy_lcs)
                                total = len(noisy_lc_list) + len(bts_lc_list)
                                this_round = 1 + len(noisy_lcs)
                                generated.update({c: generated[c] + this_round})
                                if plot_debug:
                                    for i, noisy_table in enumerate(noisy_lcs):
                                        ax = plot.plot_lc(bts_lc, noisy_table)
                                        plt.savefig(
                                            self.train_dir
                                            / f"{bts_lc.meta['name']}_{i}.pdf",
                                            format="pdf",
                                            bbox_inches="tight",
                                        )
                                        plt.close()

                            else:
                                failed["no_lc_after_cuts"].append(header.get("name"))
                else:
                    failed["no_class"].append(header.get("name"))
            else:
                failed["no_z"].append(header.get("name"))

        lc_list = [*bts_lc_list, *noisy_lc_list]

        self.logger.info(
            f"{len(failed['no_z'])} items: no redshift | {len(failed['no_lc_after_cuts'])} items: lc does not survive cuts | {len(failed['no_class'])} items: no classification"
        )

        self.logger.info(
            f"Generated {len(noisy_lc_list)+len(bts_lc_list)} lightcurves from {len(bts_lc_list)} original lightcurves"
        )
        self.logger.info(f"Kept {len(validation_lc_list)} lightcurves for validation")

        self.logger.info(f"Created per class: {generated}")

        if self.output_format == "h5":

            # Save h5 files
            dataset_h5_bts = lcdata.from_light_curves(bts_lc_list)
            dataset_h5_noisy = lcdata.from_light_curves(noisy_lc_list)
            dataset_h5_combined = lcdata.from_light_curves(lc_list)

            dataset_h5_bts.write_hdf5(
                str(self.train_dir / f"{self.name}_bts.h5"), overwrite=True
            )
            dataset_h5_noisy.write_hdf5(
                str(self.train_dir / f"{self.name}_noisy.h5"), overwrite=True
            )
            dataset_h5_combined.write_hdf5(
                str(self.train_dir / f"{self.name}_combined.h5"), overwrite=True
            )

        elif self.output_format == "csv":
            for lc in lc_list:
                io.save_csv_with_header(lc, savedir=self.train_dir)

        self.logger.info(
            f"Saved to {self.output_format} files in {self.train_dir.resolve()}"
        )
