#!/usr/bin/env python3

import logging, os, subprocess

import h5py  # type: ignore
import pandas as pd  # type: ignore


class Train:
    def __init__(self, path: str):
        self.path = path

        with h5py.File(self.path, "r") as f:
            keys = list(f.keys())
            meta = list(f["metadata"])
        meta_dict = {}
        for entry in meta:
            meta_dict.update(
                {
                    entry[0].decode("ascii"): {
                        "z": entry[4],
                        "parent_id": entry[5].decode("ascii"),
                        "parent_z": float(entry[10].decode("ascii")),
                        "class": entry[9].decode("ascii"),
                    },
                }
            )

        self.meta = pd.DataFrame.from_dict(meta_dict, orient="index")

    def print_statistics(self):
        """Get some statistics on the training set"""
        n_classes = len(unique_class := self.meta["class"].unique())
        print(f"There are {n_classes} different classes, these are:")
        for cl in unique_class:
            print(cl)
        print("--------")
        n_parent_id = len(unique_parent := self.meta["parent_id"].unique())
        print(
            f"There are {n_parent_id} parent objects. From these, {len(self.meta)} lightcurves have been created."
        )

    def run(self):
        """Run the actual train command"""


# noisification_paramdict = {
#     "max_redshift_delta": 0.1,
#     "classes": ["tde", "slsn", "baratheon", "snia"],
#     "redshift_dist": "cubic",
#     "k_scale": 1.25,
#     "phase_limit": {"long_lived": [-50, 200], "short_lived": [-30, 50]},
#     "no_k_classes": ["tde", "slsn"],
#     "sn_cuts": {"n_det": 5, "sn_threshold": 5},
#     "k_correction": None,
#     "augmentation": {
#         "mode": "per_class",
#         "multiplier": {"snia": 10, "slsn": 100, "tde": 100, "baratheon": 100},
#     },
#     "verification_percentage": 0.1,
# }

# parsnip extra parameters
# asdf
