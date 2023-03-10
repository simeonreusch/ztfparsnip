#!/usr/bin/env python3

import logging
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Tuple

import astropy  # type: ignore
import lcdata  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import parsnip  # type: ignore
import torch
from matplotlib import pyplot as plt  # type: ignore
from numpy.random import default_rng
from tqdm import tqdm
from ztfparsnip import io


class Train:
    def __init__(
        self,
        data_dir: Path | str | None = None,
        training_path: Path | str | None = None,
        classkey: str | None = None,
        no_redshift: bool = False,
        train_test_fraction: float = 0.7,
        seed=None,
        name: str = "train",
    ):
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.no_redshift = no_redshift
        self.train_test_fraction = train_test_fraction
        self.rng = default_rng(seed=seed)

        self.config = io.load_config()

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            self.device = "cuda"
            self.threads = torch.cuda.device_count()
        else:
            self.device = "cpu"
            self.threads = 8

        classkeys_available = [
            key
            for key in list(self.config.keys())
            if key not in ["sncosmo_templates", "test_lightcurves"]
        ]

        if classkey is None:
            raise ValueError(
                f"Specify a set of classifications to choose from the config. Available: {classkeys_available}"
            )
        else:
            self.classkey = classkey

        if data_dir is not None:
            self.base_dir = Path(data_dir)

        else:
            self.base_dir = io.BASE_DIR.absolute()

        if training_path is not None:
            self.training_path = Path(training_path)
        else:
            training_path = self.base_dir / "train"
            self.training_path = training_path / f"{self.name}_bts_all.h5"

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
            f"There are {n_classes} different classes in the training data, these are: {unique_class}"
        )
        n_parent_id = len(unique_parent := self.meta["parent_id"].unique())
        self.logger.info(
            f"There are {n_parent_id} parent objects. From these, {len(self.meta)} lightcurves have been created."
        )

    def run(self, threads: int = 10, model_path: str | Path | None = None):
        """Run the actual train command"""
        if model_path is not None:
            self.model_path = Path(model_path)

        else:
            model_dir = self.base_dir / "models"
            if not model_dir.exists():
                os.makedirs(model_dir)
            self.model_path = model_dir / (self.training_path.stem + "_model.hd5")

        self.logger.info(f"Running training. Outfile will be {self.model_path}")

        start_time = time.time()

        args = {
            "model_version": 2,
            "input_redshift": True,
            "predict_redshift": False,
            "specz_error": 0.01,
            "min_wave": 1000.0,
            "max_wave": 11000.0,
            "spectrum_bins": 300,
            "max_redshift": 4.0,
            "band_oversampling": 51,
            "time_window": 300,
            "time_pad": 100,
            "time_sigma": 20.0,
            "color_sigma": 0.3,
            "magsys": "ab",
            "error_floor": 0.01,
            "zeropoint": 25.0,
            "batch_size": 128,
            "learning_rate": 0.001,
            "scheduler_factor": 0.5,
            "min_learning_rate": 1e-05,
            "penalty": 0.001,
            "optimizer": "Adam",
            "sgd_momentum": 0.9,
            "latent_size": 3,
            "encode_block": "residual",
            "encode_conv_architecture": [40, 80, 120, 160, 200, 200, 200],
            "encode_conv_dilations": [1, 2, 4, 8, 16, 32, 64],
            "encode_fc_architecture": [200],
            "encode_time_architecture": [200],
            "encode_latent_prepool_architecture": [200],
            "encode_latent_postpool_architecture": [200],
            "decode_architecture": [40, 80, 160],
        }

        args["overwrite"] = True
        args["max_epochs"] = 1000

        label_map, valid_classes = self.get_classes()

        dataset = parsnip.load_datasets(
            [str(self.training_path.resolve())],
            require_redshift=False,
            label_map=label_map,
            valid_classes=valid_classes,
            kind="ztf",
        )

        bands = parsnip.get_bands(dataset)

        self.logger.info(f"Device: {self.device} / threads: {self.threads}")

        model = parsnip.ParsnipModel(
            path=str(self.model_path),
            bands=bands,
            device=self.device,
            threads=self.threads,
            settings=args,
            ignore_unknown_settings=True,
        )

        dataset = model.preprocess(dataset)

        # do own test train validation split here!!!!
        train_dataset, test_dataset = self.split_train_test(
            dataset, train_test_fraction=self.train_test_fraction
        )

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
                f"{self.model_path} {model.epoch} {elapsed_time:.2f} {train_score:.4f} "
                f"{test_score:.4f}",
                file=f,
            )

        self.logger.info(
            f"model path={self.model_path} epoch={model.epoch} elapsed_time={elapsed_time:.2f} train_score={train_score:.4f} test_score={test_score:.4f}"
        )

    def get_classes(self):
        """
        Convert classification mapping dictionary to something
        parsnip can parse (basically invert keys and values)
        """
        label_map = self.config[self.classkey]
        label_map_parsnip = {}

        train_on = label_map.get("train_on")
        if train_on is not None:
            valid_classes = label_map.pop("train_on")

        else:
            valid_classes = []
        for k, v in label_map.items():
            for entry in v:
                label_map_parsnip.update({entry: k})
                if "sn" in k:
                    label_map_parsnip.update({f"SN{entry}": k})
                    label_map_parsnip.update({f"SN {entry}": k})
            if train_on is None:
                valid_classes.append(k)

        return label_map_parsnip, valid_classes

    def split_train_test(
        self, dataset: lcdata.dataset.Dataset, train_test_fraction: float = 0.7
    ) -> Tuple[lcdata.dataset.Dataset, lcdata.dataset.Dataset]:
        """
        Split train and test set.
        Default ratio 0.1 (90% training, 10% testing)
        """
        self.logger.info(
            f"Splitting train and test dataset. Selected ratio: {train_test_fraction}"
        )
        unique_parents = self.meta.parent_id.unique()
        size_train = int(train_test_fraction * len(unique_parents))
        train_parents = self.rng.choice(unique_parents, size=size_train, replace=False)
        test_df = self.meta.query("parent_id not in @train_parents")

        self.logger.info(
            f"Making sure no parent IDs are shared between train and test. Effective ratio: {1-(len(test_df)/len(self.meta)):.2f}"
        )

        train_mask = np.ones(len(dataset), dtype=bool)
        train_mask[test_df.index.values] = False

        test_mask = ~train_mask

        train_dataset = dataset[train_mask]
        test_dataset = dataset[test_mask]

        self.logger.info(
            f"Train size: {len(train_dataset)} / test size: {len(test_dataset)}"
        )

        return train_dataset, test_dataset

    def classify(
        self,
        model_path: Path | str | None = None,
        predictions_path: Path | str | None = None,
        validation_path: Path | str | None = None,
        plot_path: Path | str | None = None,
    ):
        """
        Evaluate the trained model with the validation lightcurves
        """
        if model_path is not None:
            self.model_path = Path(model_path)

        if model_path is None:
            model_dir = self.base_dir / "models"
            if not model_dir.exists():
                os.makedirs(model_dir)
            self.model_path = model_dir / (self.training_path.stem + "_model.hd5")

        if predictions_path is not None:
            self.predictions_path = Path(predictions_path)

        if predictions_path is None:
            predictions_dir = self.base_dir / "predictions"
            if not predictions_dir.is_dir():
                os.makedirs(predictions_dir)
            self.predictions_path = predictions_dir / (
                str(self.training_path.stem) + "_predictions.h5"
            )
        if validation_path is None:
            self.validation_path = (
                self.base_dir / "validation" / (self.name + "_bts" + "_validation.h5")
            )
        else:
            self.validation_path = Path(validation_path)

        if plot_path is None:
            plot_dir = self.base_dir / "plot"
            if not plot_dir.is_dir():
                os.makedirs(plot_dir)
            self.plot_path = plot_dir / (self.name + "_bts" + "_confusion_matrix.pdf")
        else:
            self.plot_path = Path(plot_path)

        self.logger.info(
            f"Paths chosen\nInput\nModel = {self.model_path}\nValidation: {self.validation_path}\nOutput\nPredictions: {self.predictions_path}\nPlots: {self.plot_path}"
        )

        files_to_check = [self.validation_path, self.model_path]
        self.check_files(paths=files_to_check)

        model = parsnip.load_model(
            str(self.model_path),
            device=self.device,
            threads=self.threads,
        )
        self.logger.info(f"Loaded model. Parameters:\n{model}")

        label_map, valid_classes = self.get_classes()

        if not self.predictions_path.is_file():
            dataset = parsnip.load_datasets(
                [str(self.training_path.resolve())],
                require_redshift=False,
                label_map=label_map,
                valid_classes=valid_classes,
                kind="ztf",
            )

            # Parse the dataset in chunks. For large datasets, we can't fit them all in memory
            # at the same time.
            if isinstance(dataset, lcdata.HDF5Dataset):
                chunk_size = 1000
                num_chunks = dataset.count_chunks(chunk_size)
                chunks = dataset.iterate_chunks(chunk_size)
            else:
                chunks = [dataset]

            predictions_list = []

            for chunk in chunks:
                # Preprocess the light curves
                chunk = model.preprocess(chunk, verbose=False)

                # Generate the prediction
                chunk_predictions = model.predict_dataset(chunk)
                predictions_list.append(chunk_predictions)

            predictions = astropy.table.vstack(predictions_list, "exact")

            predictions.write(
                self.predictions_path,
                overwrite=True,
                serialize_meta=True,
                path="/predictions",
            )

            self.logger.info(f"Predictions written to {self.predictions_path}")

        else:
            predictions = astropy.io.misc.hdf5.read_table_hdf5(
                str(self.predictions_path)
            )

        classifier = parsnip.Classifier()

        classifier.train(predictions)

        dataset_validation = parsnip.load_datasets(
            [str(self.validation_path)],
            require_redshift=False,
            label_map=label_map,
            valid_classes=valid_classes,
            kind="ztf",
        )

        validation_predictions = model.predict_dataset(dataset_validation)

        classifications_validation = classifier.classify(validation_predictions)

        ax = parsnip.plot_confusion_matrix(
            validation_predictions, classifications_validation
        )
        plt.savefig(self.plot_path)

    def check_files(self, paths: list[Path]):
        """
        Check each path in the paths list if the file exists.
        """
        for path in paths:
            if not path.is_file():
                raise ValueError(f"{path} is not a file")
