#!/usr/bin/env python3
# Author: Lightcurve creation code by Alice Townsend (alice.townsend@hu-berlin.de)
# License: BSD-3-Clause

import os, numpy, logging

from ztfparsnip import io


class CreateLightcurves(object):
    """
    This is the parent class for the ZTF nuclear transient sample"""

    def __init__(
        self,
        weights={},
        bts_baseline_dir: str | None = None,
    ):
        super(CreateLightcurves, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Creating lightcurves")
        self.weights = weights

        if bts_baseline_dir is None:
            self.bl_dir = io.BTS_LC_BASELINE_DIR
        else:
            self.lc_dir = bts_baseline_dir
