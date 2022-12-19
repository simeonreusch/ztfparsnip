#!/usr/bin/env python3

import logging, os, subprocess

import h5py  # type: ignore
import lcdata  # type: ignore
import pandas as pd  # type: ignore


class Train:
    def __init__(self, path: str):
        self.training_path = path

        # with h5py.File(self.training_path, "r") as f:
        #     keys = list(f.keys())
        #     meta = list(f["metadata"])

        meta_df = lcdata.read_hdf5(self.training_path, in_memory=False).meta.to_pandas()

        meta_df.drop(
            columns=[
                col
                for col in list(meta_df.keys())
                if col not in ["object_id", "redshift", "name", "bts_class", "bts_z"]
            ],
            inplace=True,
        )
        meta_df["name"] = meta_df["name"].str.decode("utf-8")
        meta_df["bts_class"] = meta_df["bts_class"].str.decode("utf-8")
        meta_df["bts_z"] = meta_df["bts_z"].str.decode("utf-8").astype(float)

        meta_df.rename(
            columns={
                "name": "parent_id",
                "bts_class": "class",
                "redshift": "z",
                "bts_z": "parent_z",
            },
            inplace=True,
        )

        self.meta = meta_df

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

    def run(self, threads: int = 10, outfile: str | None = None):
        """Run the actual train command"""
        model_path = os.path.basename(self.training_path)
        print(model_path)
        # p = subprocess.Popen(
        #     f"parsnip_train --split_train_test --threads={threads}  model/model-1-50pct-sn-mw.h5 data/model-1-50-sn.h5",
        #     stdout=subprocess.PIPE,
        #     shell=True,
        # )
        # print(p.communicate())


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
