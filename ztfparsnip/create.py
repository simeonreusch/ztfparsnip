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
            self.lc_dir = io.BTS_LC_BASELINE_DIR
        else:
            self.lc_dir = bts_baseline_dir

        self.ztfids = io.get_all_ztfids(lc_dir=self.lc_dir)
        self.config = io.load_config()

    def get_lightcurves(self):
        """
        Generator to read a dataframes and headers
        """
        for ztfid in self.ztfids:
            lc = io.get_ztfid_dataframe(ztfid=ztfid, lc_dir=self.lc_dir)
            header = io.get_ztfid_header(ztfid=ztfid, lc_dir=self.lc_dir)
            yield lc, header
