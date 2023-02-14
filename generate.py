#!/usr/bin/env python3

import logging

from ztfparsnip.create import CreateLightcurves
from ztfparsnip import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

weights = {"sn_ia": 9400, "tde": 9400, "sn_other": 9400, "agn": 9400, "star": 9400}

if __name__ == "__main__":
    sample = CreateLightcurves(
        output_format="ztfnuclear",
        classkey="simpleclasses",
        weights=weights,
        train_dir="train",
        plot_dir="plot",
        seed=0,
        phase_lim=False,
        k_corr=True,
    )
    sample.select()
    # sample.create(plot_debug=False, start=3356)
    sample.create(plot_debug=False)
