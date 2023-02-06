#!/usr/bin/env python3
# Author: Lightcurve creation code by Alice Townsend (alice.townsend@hu-berlin.de)
# License: BSD-3-Clause

import os, numpy, logging


class LightcurveCreator:
    """
    Class to read in BTS sample and create a set of augmented lightcurves
    """

    def __init__(self, weights: None | dict):
        super(LightcurveCreator, self).__init__()
        self.weights = weights

    print(self.weights)
