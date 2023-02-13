#!/usr/bin/env python3
# Author: Alice Townsend (alice.townsend@hu-berlin.de)
# License: BSD-3-Clause

import os, numpy, logging, re, os, glob

from pathlib import Path

import numpy as np
from numpy.random import default_rng
import numpy.ma as ma
import pandas as pd
import sncosmo
import lcdata

from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
from copy import copy

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
        seed: int = None,
        output_format: str = "parsnip",
        delta_z: float = 0.1,
        SN_threshold: float = 5.0,
        n_det_threshold: float = 5,
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
        self.SN_threshold = SN_threshold
        self.n_det_threshold = n_det_threshold
        self.rng = default_rng(seed=self.seed)

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
        res_counter = 0
        n_iter = 0

        while len(noisy_lcs) < (self.multiplier - 1):
            new_table_list, sim_z_list = self.get_noisified_data(table, self.delta_z)

            # Add k correction
            if self.k_corr:
                delta_m_list, delta_f_list = self.get_k_correction(table, sim_z_list)

                if delta_m_list != None:
                    for i in range(len(new_table_list)):
                        new_table_list[i]["magpsf"] = (
                            new_table_list[i]["magpsf"].data + delta_m_list[i]
                        )
                        new_table_list[i]["flux"] = (
                            new_table_list[i]["flux"].data + delta_f_list[i]
                        )

            for new_table in new_table_list:
                peak_idx = np.argmax(new_table["flux"])
                sig_noise_df = pd.DataFrame(
                    data={
                        "SN": np.abs(np.array(new_table["flux"] / new_table["fluxerr"]))
                    }
                )
                count_sn = sig_noise_df[sig_noise_df["SN"] > self.SN_threshold].count()
                if (
                    new_table["flux"][peak_idx] / new_table["fluxerr"][peak_idx]
                ) > self.SN_threshold:
                    if count_sn[0] >= self.n_det_threshold:
                        noisy_lcs.append(new_table)
                        res.append(1)
                        res_counter += 1
                    else:
                        res.append(0)
                else:
                    res.append(0)

            """
            Prevent being stuck with a lightcurve never yielding a noisified one making the snt threshold. If it fails 50 times, we move on
            """
            if n_iter == 50:
                if sum(res[-50:]) == 0:
                    return None, None

            n_iter += 1

        # print(noisy_lcs[0])
        if self.output_format == "ztfnuclear":
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
            sigmapsf = sigmapsf[mask_phase]
            fid = fid[mask_phase]

        phot = {
            "jd": jd,
            "magpsf": magpsf,
            "sigmapsf": sigmapsf,
            "fid": fid,
            "magzp_orig": magzp_orig,
        }

        phot_tab = Table(phot, names=phot.keys(), meta=self.header)
        if len(phot_tab) < 1:
            return None

        phot_tab["band"] = "ztfband"

        for fid, fname in zip([1, 2, 3], ["ztfg", "ztfr", "ztfi"]):
            phot_tab["band"][phot_tab["fid"] == fid] = fname

        phot_tab["flux"] = 10 ** (-(phot_tab["magpsf"] - 25) / 2.5)
        phot_tab["fluxerr"] = np.abs(
            phot_tab["flux"] * (-phot_tab["sigmapsf"] / 2.5 * np.log(10))
        )
        # Add 7% to the errors in flux!!
        phot_tab["fluxerr"] = phot_tab["fluxerr"] * 1.07
        phot_tab["sigmapsf"] = np.abs(
            (2.5 * np.log(10)) * (phot_tab["fluxerr"] / phot_tab["flux"])
        )
        phot_tab["zp"] = 25
        phot_tab["zpsys"] = "ab"
        phot_tab.meta["z"] = self.header["bts_z"]
        phot_tab.meta["type"] = self.header["bts_class"]
        phot_tab.sort("jd")

        return phot_tab

    def get_noisified_data(self, lc_table, delta_z):
        this_lc = copy(lc_table)
        this_lc = this_lc[this_lc["flux"] > 0.0]
        if len(this_lc) == 0:
            return Table()
        mag = this_lc["magpsf"]
        magzp_orig = this_lc["magzp_orig"]
        flux_old = this_lc["flux"]
        fluxerr_old = this_lc["fluxerr"]
        truez = float(this_lc.meta["bts_z"])
        zp = this_lc["zp"]

        max_z = truez + delta_z
        z_list = self.rng.power(4, 10000) * max_z
        z_list = z_list[z_list > truez]
        noisy_lc_list = []
        z_list_update = []

        new_z = self.rng.choice(z_list)

        delta_m = cosmo.distmod(new_z) - cosmo.distmod(truez)
        flux_new = 10 ** ((25 - mag - delta_m.value) / 2.5)
        scale = (
            cosmo.luminosity_distance(truez) ** 2
            / cosmo.luminosity_distance(new_z) ** 2
        )
        df_f_old = fluxerr_old / flux_old
        df_f_new = 1 / scale * df_f_old

        flux_obs = flux_new + self.rng.normal(scale=np.sqrt(flux_new))
        fluxerr_obs = flux_new * df_f_new

        mag_new = self.flux_to_mag(flux_obs.value, zp).value
        magerr_new = np.abs((2.5 * np.log(10)) * (fluxerr_obs / flux_obs)).value
        jd_new = this_lc["jd"].data
        band_new = this_lc["band"].data
        zp_new = this_lc["zp"].data
        zpsys_new = this_lc["zpsys"].data

        if len(mag_new) > 0:
            phot = {
                "jd": jd_new,
                "flux": flux_obs,
                "fluxerr": fluxerr_obs,
                "magpsf": mag_new,
                "sigmapsf": magerr_new,
                "magzp_orig": magzp_orig,
                "band": band_new,
                "zp": zp_new,
                "zpsys": zpsys_new,
            }
            new_lc = Table(
                phot,
                names=phot.keys(),
                meta=this_lc.meta,
            )
            new_lc.meta["z"] = new_z
            noisy_lc_list.append(new_lc)
            z_list_update.append(new_z)

        return noisy_lc_list, z_list_update

    def get_k_correction(self, lc_table, z_list):
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

        kcorr_mag_list = []
        kcorr_flux_list = []

        for z_sim in z_list:
            # get the simulation flux and find k-correction mag
            model["z"] = z_sim
            bandflux_sim = model.bandflux(lc_table["band"], time=lc_table["jd"])
            bandflux_sim[bandflux_sim == 0.0] = 1e-10
            # kcorr_mag_list.append(-2.5 * np.log10(bandflux_obs / bandflux_sim))
            kcorr_mag = np.nan_to_num(self.flux_to_mag(bandflux_obs / bandflux_sim, 0))
            kcorr_mag_list.append(kcorr_mag)
            kcorr_flux_list.append(bandflux_obs - bandflux_sim)

        return kcorr_mag_list, kcorr_flux_list

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
