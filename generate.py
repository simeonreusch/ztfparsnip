#!/usr/bin/env python3

import logging
from pathlib import Path

from ztfparsnip import io
from ztfparsnip.create import CreateLightcurves
from ztfparsnip.train import Train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# weights = {"sn_ia": 9400, "tde": 9400, "sn_other": 9400, "agn": 9400, "star": 9400}
weights = {"sn_ia": 15650, "tde": 15650, "sn_other": 15650, "agn": 15650, "star": 15650}

if __name__ == "__main__":
    # sample = CreateLightcurves(
    #     output_format="parsnip",
    #     classkey="simpleclasses",
    #     weights=weights,
    #     train_dir=Path("train_parsnip_fixstar"),
    #     plot_dir=Path("plot"),
    #     validation_dir=Path("validation_parsnip_fixstar"),
    #     seed=0,
    #     phase_lim=False,
    #     k_corr=True,
    #     validation_fraction=0.0,
    # )
    # sample.select()
    # sample.create(
    #     plot_debug=False, subsampling_rate=0.9, jd_scatter_sigma=0.03, start=0
    # )
    # 4731
    train = Train(
        path=Path("train_bts_all.h5"),
        classkey="simpleclasses",
        train_test_fraction=0.7,
        no_redshift=False,
        seed=0,
    )
    # train.classify(model_path=Path("models") / "train_bts_all_model_with_z.hd5")
    # train.evaluate()
    train.run()
