# ztfparsnip
[![PyPI version](https://badge.fury.io/py/ztfparsnip.svg)](https://badge.fury.io/py/ztfparsnip)
[![CI](https://github.com/simeonreusch/ztfparsnip/actions/workflows/ci.yaml/badge.svg)](https://github.com/simeonreusch/ztfparsnip/actions/workflows/ci.yaml)
[![Coverage Status](https://coveralls.io/repos/github/simeonreusch/ztfparsnip/badge.svg?branch=main)](https://coveralls.io/github/simeonreusch/ztfparsnip?branch=main)

Retrain [Parsnip](https://github.com/LSSTDESC/parsnip) for ZTF. This is achieved by using [fpbot](https://github.com/simeonreusch/fpbot) forced photometry lightcurves of the [Bright Transient Survey](https://sites.astro.caltech.edu/ztf/bts/bts.php). These are augmented (redshifted, noisifed and - when possible - K-corrected).

The package is maintained by [A. Townsend](https://github.com/aotownsend) (HU Berlin) and [S. Reusch](https://github.com/simeonreusch) (DESY).

## Usage
### Create augmented training sample
```python
from pathlib import Path
from ztfparsnip.create import CreateLightcurves
weights = {"sn_ia": 9400, "tde": 9400, "sn_other": 9400, "agn": 9400, "star": 9400}
if __name__ == "__main__":
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
```python
from ztfparsnip.train import Train
if __name__ == "__main__":
    train = Train(classkey="simpleclasses", seed=None)
    train.run()
```

### Evaluate
Coming soon.