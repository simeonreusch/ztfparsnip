#!/usr/bin/env python3

import json
from pathlib import Path

import lcdata

dataset = lcdata.read_hdf5("train_parsnip_fixstar/train_bts_all.h5", in_memory=False)

with open("split.json", "r") as f:
    split_dict = json.load(f)


test_lcs = []
train_lcs = []

for lc in dataset.light_curves:
    ztfid = lc.meta["name"]
    if ztfid in split_dict["trainvalidation"]:
        trainvalidation_lcs.append(lc)
    elif ztfid in split_dict["test_parentonly"]:
        test_lcs.append(lc)

dataset_trainvalidation = lcdata.from_light_curves(trainvalidation_lcs)
dataset_test = lcdata.from_light_curves(test_lcs)

dataset_trainvalidation.write_hdf5(path="train_bts_all.h5", overwrite=True)
dataset_test.write_hdf5(path="test_bts.h5", overwrite=True)
