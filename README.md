# ztfparsnip
Retrain Parsnip for ZTF. This is achieved by using [fpbot](https://github.com/simeonreusch/fpbot) forced photometry lightcurves of the [Bright Transient Survey](https://sites.astro.caltech.edu/ztf/bts/bts.php). These are augmented (redshifted, noisifed and - when possible - K-corrected).

## Usage
### Augment
```python
from ztfparsnip.create import CreateLightcurves
sample = CreateLightcurves()
```

### Train
Run `ztftrain INFILE` where `INFILE` points to a `.h5` object.

### Evaluate
Coming soon.