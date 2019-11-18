#!/bin/sh

python train.py with configs/toy_config.json
python train.py with configs/toy_config.json "bayesian"=False
python test.py with configs/toy_config.json
python test.py with configs/toy_config.json "bayesian"=False

