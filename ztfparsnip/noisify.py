#!/usr/bin/env python3
# Author: Alice Townsend (alice.townsend@hu-berlin.de)
# License: BSD-3-Clause

import glob
import logging
import os
import re
from copy import copy
from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd
from numpy.random import default_rng

import lcdata  # type:ignore
import sncosmo  # type:ignore
from astropy.cosmology import Planck18 as cosmo  # type:ignore
from astropy.table import Table  # type:ignore
from ztfparsnip import io


"""
IDEAS FOR FUTURE VERSIONS:
- add random observation time "wiggles" to K-correction
- random datapoint dropout (sparsify augmented sample) 
    -> that should be included if training without noise (for a fair comparison between "standard" augmentation and noisification)
- 
"""


class Noisify(object):
    """docstring for Noisify"""

    def __init__(
        self,
        table: Table,
        header: dict,
        multiplier: int,
        remove_poor_conditions: bool = True,
        phase_lim: bool = True,
        k_corr: bool = True,
        seed: int | None = None,
        output_format: str = "parsnip",
        delta_z: float = 0.1,
        sig_noise_cut: bool = True,
        SN_threshold: float = 5.0,
        n_det_threshold: int = 5,
        detection_scale: float = 0.5,
        subsampling_rate: float = 1.0,
        jd_scatter_sigma: float = 0.0,
    ):
        super(Noisify, self).__init__()
        self.table = table
        self.header = header
        self.multiplier = multiplier
        self.remove_poor_conditions = remove_poor_conditions
        self.phase_lim = phase_lim
        self.k_corr = k_corr
        self.seed = seed
        self.output_format = output_format
        self.delta_z = delta_z
        self.sig_noise_cut = sig_noise_cut
        self.SN_threshold = SN_threshold
        self.n_det_threshold = n_det_threshold
        self.detection_scale = detection_scale
        self.subsampling_rate = subsampling_rate
        self.jd_scatter_sigma = jd_scatter_sigma
        self.rng = default_rng(seed=self.seed)
        # self.z_valid_list = self.rng.uniform(float(self.header["bts_z"]), float(self.header["bts_z"])+self.delta_z, size=10000)
        self.z_list = self.rng.power(4, 10000) * (
            float(self.header["bts_z"]) + self.delta_z
        )
        self.z_valid_list = self.z_list[self.z_list > float(self.header["bts_z"])]

    def noisify_lightcurve(self):
        """
        Noisify a lightcurve generated in create.py
        """
        noisy_lcs = []

        table = self.get_astropy_table()

        if table is None:
            return None, None

        # -------- Noisification -------- #
        res = []
        n_iter = 0

        while len(noisy_lcs) < (self.multiplier - 1):
            # check for stars
            truez = float(table.meta["bts_z"])
            if truez == 0:
                break

            new_table, sim_z = self.get_noisified_data(table)

            if new_table is not None:
                # Add k correction
                if self.k_corr:
                    delta_m, delta_f = self.get_k_correction(table, sim_z)
                    if delta_m is not None:
                        new_table["magpsf"] = new_table["magpsf"].data + delta_m
                        new_table["flux"] = new_table["flux"].data + delta_f
                # remove negative flux values
                neg_mask = new_table["flux"].data > 0.0
                new_table = new_table[neg_mask]
                if len(new_table) == 0:
                    res.append(0)
                else:
                    # Add cut on S/N
                    if self.sig_noise_cut:
                        peak_idx = np.argmax(new_table["flux"])
                        sig_noise_df = pd.DataFrame(
                            data={
                                "SN": np.abs(
                                    np.array(new_table["flux"] / new_table["fluxerr"])
                                )
                            }
                        )
                        count_sn = sig_noise_df[
                            sig_noise_df["SN"] > self.SN_threshold
                        ].count()
                        if (
                            new_table["flux"][peak_idx] / new_table["fluxerr"][peak_idx]
                        ) > self.SN_threshold:
                            if count_sn[0] >= self.n_det_threshold:
                                # Remove data points according to density distribution
                                new_idx = self.drop_points(
                                    new_table["jd"],
                                    new_table["band"],
                                    cadence_scale=self.detection_scale,
                                )
                                aug_table = new_table[new_idx]
                                # Then randomly remove datapoints, retaining (subsampling_rate)% of lc
                                if (
                                    self.subsampling_rate < 1.0
                                    and len(aug_table["flux"]) > 10
                                ):
                                    """
                                    if self.multiplier < 10.:
                                        subsampling_val = 0.9
                                    elif self.multiplier < 50.:
                                        subsampling_val = 0.8
                                    elif self.multiplier < 300.:
                                        subsampling_val = 0.7
                                    else:
                                        subsampling_val = self.subsampling_rate
                                    subsampled_length = int(
                                        len(new_table["flux"]) * subsampling_val
                                    )
                                    """
                                    subsampled_length = int(
                                        len(aug_table["flux"]) * self.subsampling_rate
                                    )
                                    indices_to_keep = self.rng.choice(
                                        len(aug_table["flux"]),
                                        subsampled_length,
                                        replace=False,
                                    )
                                    aug_table = aug_table[indices_to_keep]
                                    if self.jd_scatter_sigma > 0:
                                        aug_table = self.scatter_jd(
                                            table=aug_table, sigma=self.jd_scatter_sigma
                                        )
                                    noisy_lcs.append(aug_table)
                                else:
                                    noisy_lcs.append(aug_table)
                                res.append(1)
                            else:
                                res.append(0)
                        else:
                            res.append(0)
                    else:
                        res.append(1)
                        noisy_lcs.append(new_table)
            else:
                res.append(0)
            """
            Prevent being stuck with a lightcurve never yielding a noisified one making the snt threshold. If it fails 50 times after start or 2000 times in a row, we move on.
            """
            n_iter += 1

            if n_iter == 50 or n_iter % 2000 == 0:
                if sum(res[-50:]) == 0 or sum(res[-2000:]) == 0:
                    print(
                        f"ABORT! (stats: n_iter: {n_iter} / generated: {len(noisy_lcs)})"
                    )
                    print(table.meta["type"])
                    break

        # Augment original BTS table: remove data points according to density distribution
        idx = self.drop_points(
            table["jd"], table["band"], cadence_scale=self.detection_scale
        )
        table = table[idx]

        if self.output_format == "parsnip":
            table.keep_columns(
                ["jd", "flux", "fluxerr", "magpsf", "sigmapsf", "band", "zp", "zpsys"]
            )
            del table.meta["lastobs"]
            del table.meta["lastdownload"]
            del table.meta["lastfit"]
            for new_table in noisy_lcs:
                new_table.keep_columns(
                    [
                        "jd",
                        "flux",
                        "fluxerr",
                        "magpsf",
                        "sigmapsf",
                        "band",
                        "zp",
                        "zpsys",
                    ]
                )
                del new_table.meta["lastobs"]
                del new_table.meta["lastdownload"]
                del new_table.meta["lastfit"]

        elif self.output_format == "ztfnuclear":
            all_tables = [table]
            all_tables.extend(noisy_lcs)
            for t in all_tables:
                t["flux"] = self.convert_flux_to_original_zp(
                    t["flux"].value, t["magzp_orig"].value
                )
                t["fluxerr"] = self.convert_flux_to_original_zp(
                    t["fluxerr"].value, t["magzp_orig"].value
                )
                del t["zp"]
                t.rename_column("magzp_orig", "zp")
                t.rename_column("magzpunc_orig", "magzpunc")

        return table, noisy_lcs

    def get_astropy_table(self):
        """
        Generate astropy table from the provided lightcurve and apply
        phase limits
        """
        if self.remove_poor_conditions:
            self.table = self.table[self.table["pass"] == 1]

        jd = np.array(self.table["jd"])
        magpsf = np.array(self.table["magpsf"])
        sigmapsf = np.array(self.table["sigmapsf"])
        fid = np.array(self.table["filterid"])
        magzp_orig = np.array(self.table["magzp"])
        magzpunc_orig = np.array(self.table["magzpunc"])

        if self.phase_lim:
            if self.header["bts_class"] in [
                "SN IIn",
                "SN IIn?",
                "SLSN-I",
                "SLSN-I.5",
                "SLSN-I?",
                "SLSN-II",
                "TDE",
            ]:
                phase_min = -50
                phase_max = 200
            elif self.header["bts_class"] == "Baratheon":
                phase_min = -50
                phase_max = 365
            else:
                phase_min = -30
                phase_max = 50

            mask_phase = ((jd - float(self.header["bts_peak_jd"])) < phase_max) & (
                (jd - float(self.header["bts_peak_jd"])) > phase_min
            )

            jd = jd[mask_phase]
            magpsf = magpsf[mask_phase]
            magzp_orig = magzp_orig[mask_phase]
            magzpunc_orig = magzpunc_orig[mask_phase]
            sigmapsf = sigmapsf[mask_phase]
            fid = fid[mask_phase]

        phot = {
            "jd": jd,
            "magpsf": magpsf,
            "sigmapsf": sigmapsf,
            "fid": fid,
            "magzp_orig": magzp_orig,
            "magzpunc_orig": magzpunc_orig,
        }

        phot_tab = Table(phot, names=phot.keys(), meta=self.header)
        if len(phot_tab) < 1:
            return None

        phot_tab["band"] = "ztfband"

        for fid, fname in zip([1, 2, 3], ["ztfg", "ztfr", "ztfi"]):
            phot_tab["band"][phot_tab["fid"] == fid] = fname

        # Add 0.03 error floor to the errors in mag!! @Melissa Amenouche
        # phot_tab["sigmapsf"] = np.sqrt(phot_tab["sigmapsf"]**2 + 0.03**2)

        phot_tab["flux"] = 10 ** (-(phot_tab["magpsf"] - 25) / 2.5)
        phot_tab["fluxerr"] = np.abs(
            phot_tab["flux"] * (-phot_tab["sigmapsf"] / 2.5 * np.log(10))
        )
        phot_tab["zp"] = 25
        phot_tab["zpsys"] = "ab"
        phot_tab.meta["z"] = self.header["bts_z"]
        phot_tab.meta["type"] = self.header["bts_class"]
        phot_tab.sort("jd")

        return phot_tab

    def get_noisified_data(self, lc_table):
        this_lc = copy(lc_table)
        this_lc = this_lc[this_lc["flux"] > 0.0]
        if len(this_lc) == 0:
            return Table()
        flux_obs = this_lc["flux"]
        fluxerr_obs = this_lc["fluxerr"]
        truez = float(this_lc.meta["bts_z"])
        zp = this_lc["zp"]

        new_z = self.rng.choice(self.z_valid_list)

        d_scale = (
            cosmo.luminosity_distance(truez) ** 2
            / cosmo.luminosity_distance(new_z) ** 2
        ).value

        mask_signoise = flux_obs / fluxerr_obs < 5.0
        flux_true = flux_obs - self.rng.normal(
            scale=fluxerr_obs
        )  # get true initial flux (removing scatter)
        flux_true[mask_signoise] = flux_obs[
            mask_signoise
        ]  # but don't remove if S/N is poor

        negflux = flux_true < 0.0  # set minimum flux
        flux_true[negflux] = 0.01

        flux_z = flux_true * d_scale
        ef = 0.5425067
        eb = 5.36951258
        err_scale = (1 + ef**2) / eb**2
        fluxerr_z_obs = (
            np.sqrt(
                (
                    d_scale / (1 + 1 / (flux_true * err_scale))
                    + 1 / (1 + err_scale * flux_true)
                )
            )
            * fluxerr_obs
        )
        flux_z_obs = flux_z + self.rng.normal(scale=fluxerr_z_obs)

        zp_new = this_lc["zp"].data
        mag_new = self.flux_to_mag(flux_z_obs, zp_new)
        magerr_new = np.abs((2.5 * np.log(10)) * (fluxerr_z_obs / flux_z_obs))
        jd_new = this_lc["jd"].data
        band_new = this_lc["band"].data
        zpsys_new = this_lc["zpsys"].data
        magzp_orig = this_lc["magzp_orig"].data
        magzpunc_orig = this_lc["magzpunc_orig"].data
        fid_new = this_lc["fid"].data

        if len(mag_new) > 0:
            phot = {
                "jd": jd_new,
                "magpsf": mag_new,
                "sigmapsf": magerr_new,
                "magzp_orig": magzp_orig,
                "magzpunc_orig": magzpunc_orig,
                "fid": fid_new,
                "band": band_new,
                "flux": flux_z_obs,
                "fluxerr": fluxerr_z_obs,
                "zp": zp_new,
                "zpsys": zpsys_new,
            }
            new_lc = Table(
                phot,
                names=phot.keys(),
                meta=this_lc.meta,
            )
            new_lc.meta["z"] = new_z
            return new_lc, new_z
        else:
            return None, None

    def get_k_correction(self, lc_table, z_sim):
        # map source to a template on sncosmo
        config = io.load_config()
        type_template_map_sncosmo = config["sncosmo_templates"]

        if lc_table.meta["bts_class"] in type_template_map_sncosmo.keys():
            template = type_template_map_sncosmo[lc_table.meta["bts_class"]]
        else:
            return None, None

        if len(lc_table) == 0:
            return None, None

        source = sncosmo.get_source(template)
        model = sncosmo.Model(source=source)
        model["z"] = lc_table.meta["bts_z"]
        model["t0"] = lc_table.meta["bts_peak_jd"]

        if template == "salt2":
            model["x1"] = 1.0
            model["c"] = 0.2
            model.set_source_peakabsmag(-19.4, "bessellb", "vega")
        else:
            model["amplitude"] = 10 ** (-10)

        bandflux_obs = model.bandflux(band=lc_table["band"], time=lc_table["jd"])

        # get the simulation flux and find k-correction mag
        model["z"] = z_sim
        bandflux_sim = model.bandflux(lc_table["band"], time=lc_table["jd"])
        bandflux_sim[bandflux_sim == 0.0] = 1e-10
        kcorr_mag = np.nan_to_num(self.flux_to_mag(bandflux_obs / bandflux_sim, 0))
        kcorr_flux = bandflux_obs - bandflux_sim

        return kcorr_mag, kcorr_flux

    def scatter_jd(self, table: Table, sigma: float = 0.05) -> Table:
        """
        Add scatter to the observation jd of a table
        """
        jd_old = table["jd"].value
        jd_noise = self.rng.normal(0, sigma, len(jd_old))
        jd_scatter = jd_old + jd_noise
        table["jd"] = jd_scatter

        return table

    def drop_points(
        self, x, band, time_period: float = 5.0, cadence_scale: float = 0.5
    ):
        # Split the data based on 'band'
        band_indices = {
            "r": (band == "ztfr"),
            "g": (band == "ztfg"),
            "i": (band == "ztfi"),
        }
        # Initialise an empty list to store retained indices for each band
        retained_indices_list = []
        for band_label, indices in band_indices.items():
            # Filter x based on the current band
            band_x = x[indices]
            # Calculate the number of detections within each time period for the current band
            num_detections = np.array(
                [
                    sum(
                        (band_x >= i - time_period / 2) & (band_x < i + time_period / 2)
                    )
                    for i in band_x
                ]
            )
            # Calculate the density of detections for the current band
            density = num_detections / time_period  # len(x)
            # Drop points randomly based on the probability distribution for the current band
            random_numbers = self.rng.uniform(0, 1, len(density))
            retained_indices = [
                i
                for i, rand_num in enumerate(random_numbers)
                if rand_num < cadence_scale / density[i]
            ]
            # Add the retained indices for the current band to the list
            retained_indices_list.append(np.where(indices)[0][retained_indices])

        # Combine the retained indices for all bands
        combined_retained_indices = np.concatenate(retained_indices_list)
        # Sort the combined retained indices
        combined_retained_indices = np.sort(combined_retained_indices)
        return combined_retained_indices

    @staticmethod
    def flux_to_mag(flux, zp):
        """
        Convert flux to mag, but output nans for flux values < 0
        """
        mag = (
            -2.5 * np.log10(flux, out=np.full(len(flux), np.nan), where=(flux > 0)) + zp
        )
        return mag

    @staticmethod
    def convert_flux_to_original_zp(flux, zp):
        """
        Go from flux of zeropoint 25 to flux with original zero point
        """
        return flux * 10 ** ((-25 + zp) / 2.5)
