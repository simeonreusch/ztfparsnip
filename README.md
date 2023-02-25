# ztfparsnip
[![CI](https://github.com/simeonreusch/ztfparsnip/actions/workflows/ci.yaml/badge.svg)](https://github.com/simeonreusch/ztfparsnip/actions/workflows/ci.yaml)

Retrain [Parsnip](https://github.com/LSSTDESC/parsnip) for ZTF. This is achieved by using [fpbot](https://github.com/simeonreusch/fpbot) forced photometry lightcurves of the [Bright Transient Survey](https://sites.astro.caltech.edu/ztf/bts/bts.php). These are augmented (redshifted, noisifed and - when possible - K-corrected).

## Usage
### Create augmented training sample
```python
from pathlib import Path
from ztfparsnip.create import CreateLightcurves
weights = {"sn_ia": 9400, "tde": 9400, "sn_other": 9400, "agn": 9400, "star": 9400}
sample = CreateLightcurves(
        output_format="parsnip",
        classkey="simpleclasses",
        weights=weights,
        train_dir=Path("train"),
        plot_dir=Path("plot"),
        seed=None,
        phase_lim=True,
        k_corr=True,
    )
sample.select()
sample.create(plot_debug=False)
```

### Train Parsnip with the augmented sample
Run `ztftrain INFILE` where `INFILE` points to a `.h5` object.

### Evaluate
Coming soon.