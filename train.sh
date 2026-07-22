#!/bin/sh
set -eu

python code/src/build_feature_table.py
python code/src/train.py
