#!/usr/bin/env python3

import logging, os, subprocess, time

import lcdata  # type: ignore
import pandas as pd  # type: ignore
import numpy as np

import parsnip


class Train:
    def __init__(self, path: str):
        self.training_path = path
        self.logger = logging.getLogger(__name__)

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
        self.print_statistics()

    def print_statistics(self):
        """Get some statistics on the training set"""
        n_classes = len(unique_class := self.meta["class"].unique())
        self.logger.info(
            f"There are {n_classes} different classes in the training data, these are:"
        )
        for cl in unique_class:
            self.logger.info(cl)
        self.logger.info("--------")
        n_parent_id = len(unique_parent := self.meta["parent_id"].unique())
        self.logger.info(
            f"There are {n_parent_id} parent objects. From these, {len(self.meta)} lightcurves have been created."
        )

    def run(self, threads: int = 10, outfile: str | None = None):
        """Run the actual train command"""
        if outfile is None:
            model_dir = os.path.dirname(self.training_path)
            outfile = os.path.join(
                model_dir,
                os.path.basename(self.training_path).split(".")[0] + "_model.hd5",
            )
        # cmd = f"parsnip_train --split_train_test --threads={threads}  {outfile} {self.training_path}"

        start_time = time.time()

        parser = parsnip.build_default_argparse("Run Parsnip")
        args = vars(parser.parse_args())
        args["overwrite"] = True
        args["max_epochs"] = 1000

        dataset = parsnip.load_datasets(
            [self.training_path],
            require_redshift=True,
        )
        bands = parsnip.get_bands(dataset)

        model = parsnip.ParsnipModel(
            outfile,
            bands,
            device="cuda",
            threads=8,
            settings=args,
            ignore_unknown_settings=True,
        )
        dataset = model.preprocess(dataset)

        # do own test train validation split here!!!!
        train_dataset, test_dataset = parsnip.split_train_test(dataset)
        model.fit(
            train_dataset, test_dataset=test_dataset, max_epochs=args["max_epochs"]
        )

        rounds = int(np.ceil(25000 / len(train_dataset)))

        train_score = model.score(train_dataset, rounds=rounds)
        test_score = model.score(test_dataset, rounds=10 * rounds)

        end_time = time.time()

        # Time taken in minutes
        elapsed_time = (end_time - start_time) / 60.0

        with open("./parsnip_results.log", "a") as f:
            print(
                f"{model_path} {model.epoch} {elapsed_time:.2f} {train_score:.4f} "
                f"{test_score:.4f}",
                file=f,
            )

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
