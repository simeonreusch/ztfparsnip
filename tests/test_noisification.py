import logging
import unittest

from ztfparsnip.create import CreateLightcurves

logging.getLogger("ztfparsnip.create").setLevel(logging.DEBUG)


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    def test_noisification(self):
        self.logger.info("\n\n Generating noisified lightcurves\n")

        weights = {"sn_ia": 5, "sn_other": 5, "tde": 5}

        sample = CreateLightcurves(
            output_format="parsnip",
            classkey="simpleclasses",
            weights=weights,
            train_dir="train",
            plot_dir="plot",
            seed=100,
            phase_lim=True,
            k_corr=True,
            test=True,
        )
        sample.select()
        sample.create(plot_debug=True)
