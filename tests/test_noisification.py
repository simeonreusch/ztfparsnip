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

        for name in ["ZTF19aapreis", "ZTF20acvmzfv"]:
            path = sample.test_dir / f"{name}.csv"
            pd.read_csv(path, comment="#")

        infile_noisified = sample.train_dir / "ZTF18aavvnzu_3.csv"
        df = pd.read_csv(infile_noisified, comment="#", index_col=0)
        df.sort_values(by=["obsmjd"], inplace=True)
        mags = df.magpsf.values
        reference_mags = [
            20.8769946171293,
            23.2297010733766,
            24.5327924093533,
            23.483461797024,
            24.817103935496,
            22.0753149522546,
            22.6535606402272,
            21.3577614903492,
            21.363927213264,
            21.3113925036992,
            20.9168143754613,
            20.5272894611439,
            19.9966237872488,
            19.8439164555649,
            20.529910457209,
            19.660455824037,
            19.5313409890858,
            19.4587169077071,
            19.5892919245916,
            20.7688632609073,
            19.4293228277489,
            19.5746926499075,
            19.5222350322337,
            20.7141412176996,
            19.7623707330004,
            19.5935373102277,
            19.7798828078211,
            20.7277867745584,
            19.6901479709895,
            19.9642109782873,
            19.9495699073606,
            20.2051962124932,
            20.0574227332912,
            20.4130322594424,
            20.3269831505036,
            21.1391046686123,
            20.5867467557501,
            21.2913840385192,
            21.6514930870363,
            20.3308610031804,
            20.2434470163134,
            20.9078429008369,
            21.2112770495247,
            20.349647746173,
            21.0740875729334,
            20.9650446097583,
            21.3386966973734,
            20.3194008576004,
            21.0654014320268,
            20.4037817088396,
            20.516024704723,
            21.6254069182172,
            21.1765215991857,
            21.6460987995769,
            20.6615965525555,
            21.7230048347192,
            21.7527330144324,
            21.6232711140658,
            22.1668943157953,
            21.9277361271447,
            21.1347042909552,
            21.6254872411587,
            22.4590982017125,
            21.3895346145094,
            21.7264244456568,
        ]

        np.testing.assert_almost_equal(df.magpsf.values, reference_mags, decimal=5)
