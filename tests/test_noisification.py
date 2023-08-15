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

        infile_noisified = sample.train_dir / "ZTF20acvmzfv_0.csv"
        df = pd.read_csv(infile_noisified, comment="#", index_col=0)
        df.sort_values(by=["obsmjd"], inplace=True)
        mags = df.magpsf.values
        reference_mags = [
            21.5554662912919,
            21.4454830439884,
            21.8994505738349,
            20.9263985678942,
            22.2441919875945,
            22.9441604035219,
            24.3677687846507,
            23.0034946020167,
            25.9923875776424,
            21.166288919165,
            22.4616703310059,
            22.2717374760252,
            24.5509257118067,
            23.0496732800403,
            22.5325265022738,
            23.083043031379,
            24.9226764999814,
            22.4398638596685,
            24.9560459080379,
            23.0811064530088,
            22.6528541704615,
            22.0410089707242,
            26.2487863438652,
            22.542455135837,
            23.2840755892981,
            22.752173165173,
            23.1591188864006,
            22.2958369409946,
            22.4748987103747,
            21.7652109311016,
            22.1834925020357,
            23.2760064545463,
            21.453782780915,
            25.51678220843,
            23.2713892378493,
            22.4579786811498,
            21.9833054036885,
            21.8312489379473,
            21.426622289548,
            24.5031277018625,
            22.3342651348803,
            21.7371085948249,
            22.0218746055628,
            24.7061790780718,
            21.459619783481,
            21.3460232109134,
            22.6347220035131,
            21.7237337133645,
            23.1358092843172,
            27.1213007475346,
            23.4535474198278,
            23.3819804238554,
            22.7669806747364,
            23.0124083490981,
            23.5403594571838,
            21.7327069075098,
            21.9332199905031,
            20.9185841026669,
            25.1858890229565,
            23.1189342052481,
            21.9472126188659,
            22.8716696465227,
            24.5013767632191,
            21.9814122216605,
            21.2565084870539,
            20.3679944533671,
            21.3277675946507,
            np.nan,
            22.6356860530262,
            22.380499468911,
            21.1454104264652,
            np.nan,
            23.9287792667212,
            22.7144795127961,
            np.nan,
            np.nan,
            26.9478312904578,
            21.4967009452911,
            np.nan,
            21.3183637981514,
            22.681821358521,
            25.0532700521693,
            25.2130698086326,
            np.nan,
            np.nan,
            22.8419969910319,
            24.1228547825223,
            23.4773016564731,
            22.6151076633357,
            25.2172629074759,
            22.412318164015,
            22.9788019123923,
            21.861195670439,
            21.6241366515817,
            21.8462644340013,
            np.nan,
            23.9132407630977,
            23.1414664546552,
            24.1204424501019,
            np.nan,
            23.6827853500566,
            23.9279341598042,
            22.7107319130554,
            23.5619355552368,
            23.197728696756,
            22.8282833060649,
            21.9200578820953,
            21.7769595656204,
            21.1370167268204,
            20.7368813080075,
            23.986405198897,
            22.9442273846738,
            22.689205239773,
            21.6512648984133,
            20.1722633877352,
            19.9777378007417,
            19.9504978218954,
            20.1807533563016,
            19.9423117020691,
            21.6568898934236,
            20.2934973700786,
            21.2205463076992,
            21.1937888586985,
            20.5388164465733,
            21.5509927059373,
            20.6865696538676,
            20.9671970532137,
            21.2904036324746,
            21.5314897812842,
            21.4211729268418,
            21.4746257152212,
            21.8108353746883,
            21.8381941635663,
            21.6536134084254,
            np.nan,
            np.nan,
            np.nan,
            22.0203223169004,
            np.nan,
            np.nan,
            22.4681709636186,
            23.2458851776153,
            23.0687297739294,
            np.nan,
            21.2550136429573,
            20.3077650319721,
            np.nan,
            20.8772950503358,
            22.5432343384685,
            23.100060745203,
            np.nan,
            22.7007291187001,
            22.8887146939695,
            22.4639737204864,
            24.6791428594043,
            22.6525460714047,
            np.nan,
            23.2442200653404,
            21.4818827190562,
            24.676457956149,
            23.6289630280494,
            22.7075383258003,
        ]

        np.testing.assert_almost_equal(df.magpsf.values, reference_mags, decimal=5)
