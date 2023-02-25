import logging
import unittest

from pathlib import Path

import pandas as pd
import numpy as np

from ztfparsnip.create import CreateLightcurves

logging.getLogger("ztfparsnip.create").setLevel(logging.DEBUG)


class TestNoisification(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def test_noisification_parsnip(self):
        self.logger.info("\n\n Generating noisified lightcurves\n")

        weights = {"sn_ia": 20, "sn_other": 20, "tde": 20}

        sample = CreateLightcurves(
            output_format="parsnip",
            classkey="simpleclasses",
            weights=weights,
            train_dir=Path("train"),
            plot_dir=Path("plot"),
            seed=100,
            phase_lim=True,
            k_corr=True,
            test=True,
        )
        sample.select()
        sample.create(plot_debug=True)

    def test_noisification_csv(self):
        self.logger.info("\n\n Generating noisified lightcurves\n")

        weights = {"sn_ia": 20, "sn_other": 20, "tde": 20}

        sample = CreateLightcurves(
            output_format="ztfnuclear",
            classkey="simpleclasses",
            weights=weights,
            train_dir=Path("train"),
            plot_dir=Path("plot"),
            seed=100,
            phase_lim=True,
            k_corr=True,
            plot_magdist=True,
            test=True,
        )
        sample.select()
        sample.create(plot_debug=True)

        for name in ["ZTF18aamvfeb", "ZTF19aapreis", "ZTF20acvmzfv"]:
            path = Path("validation") / f"{name}.csv"
            pd.read_csv(path, comment="#")

        infile_noisified = Path("train") / "ZTF18aavvnzu_3.csv"
        df = pd.read_csv(infile_noisified, comment="#", index_col=0)
        df.sort_values(by=["obsmjd"], inplace=True)
        mags = df.magpsf.values
        reference_mags = [
            25.29078394,
            25.76189095,
            27.2655531,
            24.6668269,
            24.53136655,
            np.nan,
            25.57881118,
            24.96131425,
            24.6494601,
            29.39622331,
            np.nan,
            23.98498964,
            24.12828652,
            23.1775983,
            22.00932895,
            22.55727056,
            21.62206101,
            21.39578053,
            22.05108525,
            20.94277367,
            21.03368154,
            20.34244796,
            19.9950083,
            20.78069822,
            19.88999309,
            19.75953625,
            19.52539882,
            19.73810262,
            21.16211422,
            19.56434958,
            19.8297774,
            19.58625593,
            21.16368346,
            20.15647694,
            19.85284108,
            19.89014672,
            21.39221394,
            19.99618702,
            20.10251282,
            20.18338371,
            20.46475862,
            20.27376621,
            20.45132811,
            20.75834641,
            21.78499175,
            21.01347781,
            21.36923858,
            21.61849326,
            20.41307089,
            20.23913687,
            21.45339639,
            21.27754412,
            20.53259678,
            21.24645026,
            21.23964233,
            21.81331908,
            20.51849827,
            21.42855531,
            21.92474183,
            20.7997237,
            20.86847922,
            21.26933793,
            21.63336927,
            22.15300501,
            21.05602829,
            21.85539486,
            22.23739681,
            20.93244031,
            21.78658614,
            21.84444574,
            20.98134349,
            22.26270503,
            21.43027687,
            21.87283486,
            22.18428394,
            21.40032939,
            22.04574997,
        ]

        np.testing.assert_almost_equal(df.magpsf.values, reference_mags, decimal=3)
