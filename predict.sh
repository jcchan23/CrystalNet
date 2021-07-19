#!/bin/bash
nohup python -u predict.py --gpu 0 --seed 0 --data_path ./data/material_project --test_path ./data/material_project --dataset_name band_gap --checkpoint_dir ./ckpt/ensemble_band_gap/ --no_features_scaling > ./log/predict_band_gap.log 2>&1 &
wait

