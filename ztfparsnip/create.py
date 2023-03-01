#!/usr/bin/env python3
# Author: Lightcurve creation code by Alice Townsend (alice.townsend@hu-berlin.de)
# License: BSD-3-Clause

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import lcdata  # type: ignore
import matplotlib.pyplot as plt  # type:ignore
import numpy
from numpy.random import default_rng
from tqdm import tqdm
from ztfparsnip import io, plot
from ztfparsnip.noisify import Noisify


class CreateLightcurves(object):
    """
    This is the parent class for the ZTF nuclear transient sample"""

    def __init__(
        self,
        classkey: str | None = None,
        weights: None | dict[str, Any] = None,
        validation_fraction: float = 0.1,
        k_corr: bool = True,
        seed: int | None = None,
        bts_baseline_dir: Path = io.BTS_LC_BASELINE_DIR,
        name: str = "train",
        reprocess_headers: bool = False,
        output_format: str = "parsnip",
        plot_magdist: bool = False,
        phase_lim: bool = True,
        train_dir: Path = io.TRAIN_DATA,
        plot_dir: Path = io.PLOT_DIR,
        validation_dir: Path | None | str = None,
        test: bool = False,
    ):
        super(CreateLightcurves, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Creating lightcurves")
        self.weights = weights
        self.validation_fraction = validation_fraction
        self.k_corr = k_corr
        self.seed = seed
        self.name = name
        self.phase_lim = phase_lim
        self.output_format = output_format
        self.plot_magdist = plot_magdist
        self.train_dir = train_dir
        self.plot_dir = plot_dir
        self.lc_dir = bts_baseline_dir
        self.test = test

        self.rng = default_rng(seed=self.seed)

        self.param_info = {
            "desired # of lightcurves": self.weights,
            "random seed": self.seed,
            "phase limit": self.phase_lim,
            "k correction": self.k_corr,
            "validation sample fraction": self.validation_fraction,
        }

        assert self.output_format in ["parsnip", "ztfnuclear"]

        if isinstance(self.train_dir, str):
            self.train_dir = Path(self.train_dir)
        if isinstance(self.lc_dir, str):
            self.lc_dir = Path(self.lc_dir)
        if isinstance(self.plot_dir, str):
            self.plot_dir = Path(self.plot_dir)

        if validation_dir is None:
            self.validation_dir = self.train_dir.resolve().parent / "validation"
        else:
            self.validation_dir = Path(validation_dir)

        for p in [self.train_dir, self.plot_dir, self.validation_dir]:
            if not p.exists():
                os.makedirs(p)

        self.config = io.load_config()

        """
        if we are in the default sample dir, check if files are there,
        check if files are there an download if not
        """
        if self.lc_dir == io.BTS_LC_BASELINE_DIR:
            if not self.test:
                nr_files = len([x for x in self.lc_dir.glob("*") if x.is_file()])
            else:
                nr_files = 0
                for x in self.lc_dir.glob("*"):
                    if f"{x.name}".split("_")[0] in self.config["test_lightcurves"]:
                        nr_files += 1
            if (self.test == False and nr_files < 6841) or (
                self.test and nr_files < 10
            ):
                self.logger.info("Downloading sample")
                io.download_sample(test=test)

        self.ztfids = io.get_all_ztfids(lc_dir=self.lc_dir, test=self.test)

        classkeys_available = [
            key
            for key in list(self.config.keys())
            if key not in ["sncosmo_templates", "test_lightcurves"]
        ]

        if classkey is None:
            raise ValueError(
                f"Specify a set of classifications to choose from the config. Available: {classkeys_available}"
            )
        else:
            self.classkey = classkey

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
            f"\n---------------------------------\nSelected configuration\nweights: {weights_info}\nk correction: {self.k_corr}\nvalidation fraction: {self.validation_fraction}\nseed: {self.seed}\noutput format: {self.output_format}\ntraining data output directory: {self.train_dir}\n---------------------------------"
        )

    def get_simple_class(self, classkey: str, bts_class: str) -> str:
        """
        Look in the config file to get the simple classification for a transient
        """
        for key, val in self.config[classkey].items():
            for entry in self.config[classkey][key]:
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
                simple_class = self.get_simple_class(
                    classkey=self.classkey, bts_class=bts_class
                )
                header[self.classkey] = simple_class
                yield lc, header
            else:
                yield None, header

    def get_headers(self, reprocess: bool = False):
        """
        Read headers only
        """
        headers_raw = {}
        self.headers = {}

        header_path = self.lc_dir / "headers.json"

        if not header_path.is_file() or reprocess:
            self.logger.info(
                f"Reading all lightcurves to create header json file for speedup (location: {header_path})."
            )

            for ztfid in tqdm(self.ztfids, total=len(self.ztfids)):
                _, header = io.get_lightcurve(ztfid=ztfid, lc_dir=self.lc_dir)
                header[self.classkey] = self.get_simple_class(
                    classkey=self.classkey, bts_class=header.get("bts_class")
                )
                headers_raw.update({header.get("name"): header})

            with open(header_path, "w") as f:
                f.write(json.dumps(headers_raw))

        else:
            with open(header_path, "r") as f:
                headers_raw = json.load(f)

        for k, v in headers_raw.items():
            if k in self.ztfids and (z := v.get("bts_z")) != "-" and z is not None:
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
        for c in self.config[self.classkey]:
            classes_available.update({c: {"ztfids": []}})
            for entry in self.headers.values():
                if entry.get(self.classkey) == c:
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

        available_dict = {}
        availability = ""
        for k, v in classes_available.items():
            availability += f"{k}: {classes_available[k]['entries']}\n"
            available_dict.update({k: classes_available[k]["entries"]})
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

        self.param_info.update(
            {"expected # of lightcurves": expected, "weights": self.selection}
        )
        self.param_info.update({"available lightcurves": available_dict})

        self.classes_available = classes_available

    def create(
        self,
        plot_debug: bool = False,
        sig_noise_mask: bool = True,
        start: int = 0,
        sig_noise_cut: bool = True,
        delta_z: float = 0.1,
        SN_threshold: float = 5.0,
        n_det_threshold: float = 5.0,
        subsampling_rate: float = 1.0,
        jd_scatter_sigma: float = 0.0,
        n: int | None = None,
        plot_iband: bool = False,
    ):
        """
        Create noisified lightcurves from the sample
        """
        failed: dict[str, list] = {"no_z": [], "no_class": [], "no_lc_after_cuts": []}

        final_lightcurves: dict[str, list] = {
            "bts_validation": [],
            "bts_orig": [],
            "bts_noisified": [],
        }

        self.param_info.update(
            {
                "max_redshift_delta": delta_z,
                "sn_cuts": {
                    "SN_threshold": SN_threshold,
                    "n_det_threshold": n_det_threshold,
                },
                "subsampling_rate": subsampling_rate,
                "jd_scatter_sigma": jd_scatter_sigma,
            }
        )

        generated = {k: 0 for (k, v) in self.selection.items()}

        for lc, header in self.get_lightcurves(start=start, end=n):
            if lc is not None:
                if (c := header[self.classkey]) is not None:
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
                            k_corr=self.k_corr,
                            seed=self.seed,
                            phase_lim=self.phase_lim,
                            delta_z=delta_z,
                            sig_noise_cut=sig_noise_cut,
                            SN_threshold=SN_threshold,
                            n_det_threshold=n_det_threshold,
                            subsampling_rate=subsampling_rate,
                            jd_scatter_sigma=jd_scatter_sigma,
                            output_format=self.output_format,
                        )

                        if get_validation:
                            validation_lc, _ = noisify.noisify_lightcurve()
                            if validation_lc is not None:
                                final_lightcurves["bts_validation"].append(
                                    validation_lc
                                )
                                if self.output_format == "ztfnuclear":
                                    io.save_csv_with_header(
                                        validation_lc,
                                        savedir=self.validation_dir,
                                        output_format=self.output_format,
                                    )

                        else:
                            bts_lc, noisy_lcs = noisify.noisify_lightcurve()
                            if bts_lc is not None:
                                for i, noisy_lc in enumerate(noisy_lcs):
                                    noisy_lc.meta["name"] = (
                                        noisy_lc.meta["name"] + f"_{i}"
                                    )
                                final_lightcurves["bts_orig"].append(bts_lc)
                                final_lightcurves["bts_noisified"].extend(noisy_lcs)

                                if self.output_format == "ztfnuclear":
                                    io.save_csv_with_header(
                                        bts_lc,
                                        savedir=self.train_dir,
                                        output_format=self.output_format,
                                    )
                                    for noisy_lc in noisy_lcs:
                                        io.save_csv_with_header(
                                            noisy_lc,
                                            savedir=self.train_dir,
                                            output_format=self.output_format,
                                        )

                                this_round = 1 + len(noisy_lcs)
                                generated.update({c: generated[c] + this_round})
                                if plot_debug:
                                    for noisy_lc in noisy_lcs:
                                        ax = plot.plot_lc(
                                            bts_lc,
                                            noisy_lc,
                                            phase_limit=self.phase_lim,
                                            sig_noise_mask=sig_noise_mask,
                                            output_format=self.output_format,
                                            plot_iband=plot_iband,
                                        )
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
            ax2 = plot.plot_magnitude_dist(final_lightcurves)
            plt.savefig(
                self.plot_dir / "mag_vs_magerr.pdf", format="pdf", bbox_inches="tight"
            )
            plt.close()

        self.logger.info(
            f"{len(failed['no_z'])} items: no redshift | {len(failed['no_lc_after_cuts'])} items: lc does not survive cuts | {len(failed['no_class'])} items: no classification"
        )

        self.logger.info(
            f"Generated {len(final_lightcurves['bts_noisified'])} noisified additional lightcurves from {len(final_lightcurves['bts_orig'])} original lightcurves"
        )
        self.logger.info(
            f"Kept {len(final_lightcurves['bts_validation'])} lightcurves for validation"
        )

        self.logger.info(f"Created per class: {generated}")

        self.param_info.update({"generated lightcurves": generated})

        if self.output_format == "parsnip":
            # Save h5 files
            for k, v in final_lightcurves.items():
                if len(v) > 0:
                    if k == "bts_validation":
                        output_dir = self.validation_dir
                    else:
                        output_dir = self.train_dir
                    dataset = lcdata.from_light_curves(v)
                    dataset.write_hdf5(
                        str(output_dir / f"{self.name}_{k}.h5"), overwrite=True
                    )

        self.logger.info(
            f"Saved to {self.output_format} files in {self.train_dir.resolve()}"
        )

        with open(self.train_dir / "info.json", "w") as f:
            json.dump(self.param_info, f)
