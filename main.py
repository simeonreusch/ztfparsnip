#!/usr/bin/env python3

import logging, os
import argparse

from ztfparsnip.train import Train


def run():
    logging.basicConfig(level=logging.INFO)
    _parser = argparse.ArgumentParser(
        description="Train Parsnip on noisified lightcurves"
    )

    _parser.add_argument(
        "file",
        type=str,
        help="Provide a training sample filepath",
    )
    args = _parser.parse_args()

    t = Train(args.file, seed=3)
    t.run()


if __name__ == "__main__":
    run()
