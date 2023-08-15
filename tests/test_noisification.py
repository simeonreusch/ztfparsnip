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

        infile_noisified = sample.train_dir / "ZTF20acueziy_2.csv"
        df = pd.read_csv(infile_noisified, comment="#", index_col=0)
        df.sort_values(by=["obsmjd"], inplace=True)

        mags = df.magpsf.values
        reference_mags = [
            20.7862801545962,
            19.9859143237067,
            20.3707809492468,
            19.9429214884179,
            20.4923597159272,
            19.888937399982,
            20.8460105835683,
            20.7091668978473,
            20.0292037077898,
            21.1112591814753,
            20.1875765243095,
            21.0763226188922,
            20.3086801104863,
            21.9130488366769,
            20.6240161912531,
            21.7743936386835,
            20.9042362388927,
            20.0740109279902,
            21.239673495802,
            20.7024764008374,
            25.7528842507013,
            20.4149368461691,
            21.1107872472987,
            20.8958085465419,
            22.5168452113857,
            21.2957311943224,
            20.9247303408706,
            21.1622199708444,
            22.8048906115971,
            21.3839012928751,
            np.nan,
            21.1104346878744,
            20.9656819124267,
            21.2908527288881,
            20.9777549528361,
            22.1839553755309,
            22.0672315128775,
            21.8042786749289,
            21.7824972084558,
            24.5796190172382,
        ]

        np.testing.assert_almost_equal(df.magpsf.values, reference_mags, decimal=5)
