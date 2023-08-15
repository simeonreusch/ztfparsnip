import logging
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ztfparsnip.create import CreateLightcurves

logging.getLogger("ztfparsnip.create").setLevel(logging.DEBUG)
logging.getLogger("ztfparsnip.io").setLevel(logging.DEBUG)


class TestNoisification(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def test_noisification_parsnip(self):
        self.logger.info("\n\n Generating noisified lightcurves\n")

        weights = {"snia": 20, "sn_other": 20, "tde": 20}

        sample = CreateLightcurves(
            output_format="parsnip",
            classkey="simpleclasses",
            weights=weights,
            train_dir=Path("train"),
            plot_dir=Path("plot"),
            seed=100,
            phase_lim=True,
            k_corr=True,
            testing=True,
        )
        sample.select()
        sample.create(plot_debug=True)

    def test_noisification_csv(self):
        self.logger.info("\n\n Generating noisified lightcurves\n")

        weights = {"snia": 20, "sn_other": 20, "tde": 20}

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
            testing=True,
        )
        sample.select()
        sample.create(plot_debug=True)

        available = [x for x in sample.test_dir.glob("*") if x.is_file()]

        for entry in available:
            print(entry)

        for name in ["ZTF19aapreis", "ZTF18aamvfeb"]:
            path = sample.test_dir / f"{name}.csv"
            pd.read_csv(path, comment="#")

        for entry in [x for x in sample.train_dir.glob("*") if x.is_file()]:
            print(entry)

        infile_noisified = sample.train_dir / "ZTF18aamvfeb_1.csv"
        df = pd.read_csv(infile_noisified, comment="#", index_col=0)
        df.sort_values(by=["obsmjd"], inplace=True)
        mags = df.magpsf.values
        reference_mags = [
            20.8441333516679,
            23.0428384270284,
            22.3361328560728,
            21.891844441017,
            24.6258793611999,
            22.9693164240199,
            22.5331418225072,
            22.402843093917,
            22.3810015045769,
            22.2366564452148,
            21.5414180986661,
            21.2231186286774,
            21.2711180499247,
            21.2342274051828,
            21.2176177244074,
            20.5653512827215,
            20.5539113677205,
            20.7776896403664,
            20.0722946507605,
            20.3702511077786,
            19.9008369045443,
            19.734931303138,
            19.9527261644008,
            20.2696107666537,
            19.7148050782857,
            19.7823696906336,
            19.8892495623189,
            20.015566245011,
            20.1393488749641,
            21.3747144565654,
            20.1022049890303,
            20.1660146696708,
            20.3844328005527,
            20.3033841862743,
            20.3979200335415,
            20.4386545415709,
            21.6346780770264,
            20.6075448924875,
            20.7080863740433,
            20.741039677494,
            21.5104615666065,
            21.0652872871537,
            20.7413309573481,
            21.1789265051537,
            21.3725943339692,
            21.610544017918,
            20.6746444748398,
            20.8608122584386,
            22.0703410057646,
            21.6514737163237,
            21.0137037685049,
            21.9315468152393,
            20.7816733808444,
            21.3105951387263,
            21.3935822142343,
            22.6814688354199,
            21.4360049799304,
            21.482178701566,
            21.6489462897514,
            22.7800350088775,
            21.771910283692,
            22.4665709831155,
            21.1519000027794,
            21.2051984992754,
            22.531694273597,
            21.7373170112264,
            23.0049914868813,
        ]

        np.testing.assert_almost_equal(df.magpsf.values, reference_mags, decimal=5)
