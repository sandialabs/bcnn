#!/bin/sh

rm -rf toy_data toy_weights toy_images toy_predict
python generate_toy_data.py
python train.py with configs/toy_config.json
python train.py with configs/toy_config.json "bayesian"=False
python test.py with configs/toy_config.json
python test.py with configs/toy_config.json "bayesian"=False

