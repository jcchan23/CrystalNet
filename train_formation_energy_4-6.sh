#!/bin/bash
nohup python -u train.py --gpu 0 --seed 0 --data_path ./data/material_project --train_path ./data/material_project --dataset_name formation_energy --dataset_type regression --run_fold 4 --metric mae --save_dir ./ckpt/ensemble_formation_energy --epochs 100 --init_lr 1e-4 --max_lr 3e-4 --final_lr 1e-4 --no_features_scaling --show_individual_scores > ./log/fold_4_formation_energy.log 2>&1 &
nohup python -u train.py --gpu 1 --seed 0 --data_path ./data/material_project --train_path ./data/material_project --dataset_name formation_energy --dataset_type regression --run_fold 5 --metric mae --save_dir ./ckpt/ensemble_formation_energy --epochs 100 --init_lr 1e-4 --max_lr 3e-4 --final_lr 1e-4 --no_features_scaling --show_individual_scores > ./log/fold_5_formation_energy.log 2>&1 &
nohup python -u train.py --gpu 2 --seed 0 --data_path ./data/material_project --train_path ./data/material_project --dataset_name formation_energy --dataset_type regression --run_fold 6 --metric mae --save_dir ./ckpt/ensemble_formation_energy --epochs 100 --init_lr 1e-4 --max_lr 3e-4 --final_lr 1e-4 --no_features_scaling --show_individual_scores > ./log/fold_6_formation_energy.log 2>&1 &
wait