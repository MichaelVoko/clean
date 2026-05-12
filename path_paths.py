#!/usr/bin/env python3
"""Usage: python patch_paths.py dfm_model.json /new/repo/root"""
import json, sys, os

json_path = sys.argv[1]
new_root = sys.argv[2].rstrip("/")

with open(json_path) as f:
    params = json.load(f)

params["DF_PATH_TRAIN"] = f"{new_root}/data/datasets/design_from_splits/design_train.csv"
params["DF_PATH_VALID"] = f"{new_root}/data/datasets/design_from_splits/design_evaluation_valid.csv"
params["BASE_FOLDER"] = f"{new_root}/dfm_model"
params["PREV_CHECKPOINT"] = ""
params["CHAIN_CLUSTER_LOOKUP_PATH"] = f"{new_root}/data/datasets/design_from_splits/chain_cluster_lookup.pkl"

with open(json_path, "w") as f:
    json.dump(params, f, indent=4)

print(f"Patched {json_path} with root={new_root}")
