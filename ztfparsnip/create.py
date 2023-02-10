#!/usr/bin/env python3
# Author: Lightcurve creation code by Alice Townsend (alice.townsend@hu-berlin.de)
# License: BSD-3-Clause

import os, numpy, logging, json, math

from pathlib import Path

from typing import Any
from numpy.random import default_rng

from tqdm import tqdm

from ztfparsnip import io
from ztfparsnip.noisify import Noisify
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
        seed: int | None = None,
        bts_baseline_dir: Path = io.BTS_LC_BASELINE_DIR,
        name: str = "train",
        reprocess_headers: bool = False,
        output_format: str = "parsnip",
        plot_magdist: bool = True,
        phase_lim: bool = True,
        train_dir: Path = io.TRAIN_DATA,
        plot_dir: Path = io.PLOT_DIR,
    ):
        super(CreateLightcurves, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Creating lightcurves")
        self.weights = weights
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.name = name
        self.phase_lim = phase_lim
        self.output_format = output_format
        self.plot_magdist = plot_magdist
        self.train_dir = train_dir
        self.plot_dir = plot_dir
        self.lc_dir = bts_baseline_dir

        self.rng = default_rng(seed=self.seed)

        assert self.output_format in ["parsnip", "ztfnuclear"]

        if isinstance(self.train_dir, str):
            self.train_dir = Path(self.train_dir)
        if isinstance(self.lc_dir, str):
            self.lc_dir = Path(self.lc_dir)
        if isinstance(self.plot_dir, str):
            self.plot_dir = Path(self.plot_dir)

        for p in [self.train_dir, self.plot_dir]:
            if not p.exists():
                os.makedirs(p)

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
            f"\n---------------------------------\nSelected configuration\nweights: {weights_info}\nvalidation fraction: {self.validation_fraction}\nseed: {self.seed}\noutput format: {self.output_format}\ntraining data output directory: {self.train_dir}\n---------------------------------"
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

    def create(self, plot_debug: bool = False, n: int | None = None):
        """
        Create noisified lightcurves from the sample
        """
        failed = {"no_z": [], "no_class": [], "no_lc_after_cuts": []}

        final_lightcurves = {"validation": [], "bts_orig": [], "bts_noisified": []}

        generated = {k: 0 for (k, v) in self.selection.items()}

        for lc, header in self.get_lightcurves(end=n):
            if lc is not None:
                if (c := header.get("simple_class")) is not None:
                    if c in self.selection.keys():

                        # check if it's a validation sample lightcurve
                        if header["name"] in self.validation_sample["all"]["ztfids"]:
                            multiplier = 0
                            get_validation = True
                        else:
                            multiplier = self.selection[c]
                            get_validation = False

                        noisify = Noisify(
                            table=lc,
                            header=header,
                            multiplier=multiplier,
                            seed=self.seed,
                            phase_lim=self.phase_lim,
                        )

                        if get_validation:
                            validation_lc, _ = noisify.noisify_lightcurve()
                            if validation_lc is not None:
                                final_lightcurves["validation"].append(validation_lc)

                        else:
                            bts_lc, noisy_lcs = noisify.noisify_lightcurve()
                            if bts_lc is not None:
                                for i, noisy_lc in enumerate(noisy_lcs):
                                    noisy_lc.meta["name"] = (
                                        noisy_lc.meta["name"] + f"_{i}"
                                    )
                                final_lightcurves["bts_orig"].append(bts_lc)
                                final_lightcurves["bts_noisified"].extend(noisy_lcs)
                                this_round = 1 + len(noisy_lcs)
                                generated.update({c: generated[c] + this_round})
                                if plot_debug:
                                    for noisy_lc in noisy_lcs:
                                        ax = plot.plot_lc(bts_lc, noisy_lc)
                                        plt.savefig(
                                            self.plot_dir
                                            / f"{noisy_lc.meta['name']}.pdf",
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

        final_lightcurves["bts_all"] = [
            *final_lightcurves["bts_orig"],
            *final_lightcurves["bts_noisified"],
        ]

        if self.plot_magdist:
            plot.plot_magnitude_dist(final_lightcurves)

        self.logger.info(
            f"{len(failed['no_z'])} items: no redshift | {len(failed['no_lc_after_cuts'])} items: lc does not survive cuts | {len(failed['no_class'])} items: no classification"
        )

        self.logger.info(
            f"Generated {len(final_lightcurves['bts_all'])} lightcurves from {len(final_lightcurves['bts_orig'])} original lightcurves"
        )
        self.logger.info(
            f"Kept {len(final_lightcurves['validation'])} lightcurves for validation"
        )

        self.logger.info(f"Created per class: {generated}")

        if self.output_format == "parsnip":

            # Save h5 files
            for k, v in final_lightcurves.items():
                if len(v) > 0:
                    dataset = lcdata.from_light_curves(v)
                    dataset.write_hdf5(
                        str(self.train_dir / f"{self.name}_{k}.h5"), overwrite=True
                    )

        elif self.output_format == "ztfnuclear":
            for lc in final_lightcurves["bts_all"]:
                io.save_csv_with_header(lc, savedir=self.train_dir)

        self.logger.info(
            f"Saved to {self.output_format} files in {self.train_dir.resolve()}"
        )

    def create_magdist_plot(self, n: int | None = None):
        """
        Creat a plot to show the magnitude vs magnitude error distribution for the sample (BTS vs. noisified)
        """
        failed = {"no_z": [], "no_class": [], "no_lc_after_cuts": []}

        bts_mag_list = []
        bts_magerr_list = []
        noisy_mag_list = []
        noisy_magerr_list = []

        generated = {k: 0 for (k, v) in self.selection.items()}

        for lc, header in self.get_lightcurves(end=n):
            if lc is not None:
                if (c := header.get("simple_class")) is not None:
                    if c in self.selection.keys():
                        multiplier = self.selection[c]
                        bts_lc, noisy_lcs = noisify.noisify_lightcurve(
                            lc, header, multiplier
                        )
                        if bts_lc is not None:
                            bts_mag_list.append(bts_lc["magpsf"].data)
                            bts_magerr_list.append(bts_lc["sigmapsf"].data)
                            noisy_mag_list.extend(
                                [noisy_lc["magpsf"].data for noisy_lc in noisy_lcs]
                            )
                            noisy_magerr_list.extend(
                                [noisy_lc["sigmapsf"].data for noisy_lc in noisy_lcs]
                            )
                            this_round = 1 + len(noisy_lcs)
                            generated.update({c: generated[c] + this_round})

                        else:
                            failed["no_lc_after_cuts"].append(header.get("name"))
                else:
                    failed["no_class"].append(header.get("name"))
            else:
                failed["no_z"].append(header.get("name"))

        self.logger.info(
            f"{len(failed['no_z'])} items: no redshift | {len(failed['no_lc_after_cuts'])} items: lc does not survive cuts | {len(failed['no_class'])} items: no classification"
        )

        self.logger.info(f"Created per class: {generated}")

        plot_dist = plot.plot_magnitude_dist(
            bts_mag_list, bts_magerr_list, noisy_mag_list, noisy_magerr_list
        )
        plt.savefig(
            self.train_dir / "mag_vs_magerr.pdf", format="pdf", bbox_inches="tight"
        )
