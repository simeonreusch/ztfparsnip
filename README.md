# ztfparsnip
[![PyPI version](https://badge.fury.io/py/ztfparsnip.svg)](https://badge.fury.io/py/ztfparsnip)
[![CI](https://github.com/simeonreusch/ztfparsnip/actions/workflows/ci.yaml/badge.svg)](https://github.com/simeonreusch/ztfparsnip/actions/workflows/ci.yaml)
[![Coverage Status](https://coveralls.io/repos/github/simeonreusch/ztfparsnip/badge.svg?branch=main)](https://coveralls.io/github/simeonreusch/ztfparsnip?branch=main)

Retrain [Parsnip](https://github.com/LSSTDESC/parsnip) for [ZTF](https://www.ztf.caltech.edu/). This is achieved by using [fpbot](https://github.com/simeonreusch/fpbot) forced photometry lightcurves of the [Bright Transient Survey](https://sites.astro.caltech.edu/ztf/bts/bts.php). These are augmented (redshifted, noisified and - when possible - K-corrected).

The package is maintained by [A. Townsend](https://github.com/aotownsend) (HU Berlin) and [S. Reusch](https://github.com/simeonreusch) (DESY).

The following augmentation steps are taken for each parent lightcurve to generate a desired number of children (calculated via `weights`):

- draw a new redshift from a cubic distribution with maximum redshift increase `delta_z`
- only accept the lightcurve if at least `n_det_threshold` datapoints are above the signal-to-noise threshold `SN_threshold`
- if the lightcurve has an existing SNCosmo template, apply a [K-correction](https://en.wikipedia.org/wiki/K_correction) at that magnitude (if `k_corr=True`)
- randomly drop datapoints until `subsampling_rate` is reached
- add some scatter to the observed dates (`jd_scatter_sigma` in days)
- if `phase_lim=True`, only keep datapoints during a typical duration (depends on the type of source)

:warning:
Note that a high `delta_z` without loosening the `SN_threshold` and `n_det_threshold` will result in a large dropout rate, which will ultimately lead to far less lightcurves being generated than initially desired.

## Usage
### Create an augmented training sample
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

### Evaluate the Parsnip model
```python
from ztfparsnip.train import Train

if __name__ == "__main__":
    train = Train(classkey="simpleclasses", seed=None)
    train.classify()
```
