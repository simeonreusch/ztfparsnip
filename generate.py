#!/usr/bin/env python3

import logging
from pathlib import Path

from ztfparsnip import io
from ztfparsnip.create import CreateLightcurves

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

weights = {"sn_ia": 9400, "tde": 9400, "sn_other": 9400, "agn": 9400, "star": 9400}

if __name__ == "__main__":
    sample = CreateLightcurves(
        output_format="parsnip",
        classkey="simpleclasses",
        weights=weights,
        train_dir=Path("train"),
        plot_dir=Path("plot"),
        seed=0,
        phase_lim=False,
        k_corr=True,
        validation_fraction=0.3,
    )
    sample.select()
    sample.create(plot_debug=False)
    train = Train(classkey="simpleclasses", seed=0)
    train.run()
