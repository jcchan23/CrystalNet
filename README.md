# CrystalNet

Source code for our paper [Enhancing Material Property Prediction by Leveraging Large-scale Computational Database and Machine Learning](#)

The code was built based on [CMPNN](https://github.com/SY575/CMPNN) and optimized for the crystal graph. Thanks a lot for their code sharing!

## Dependencies
+ cuda == 10.1
+ cudnn >= 7.4.1
+ pymatgen == 2020.12.18
+ torch == 1.5.0
+ numpy == 1.20.2
+ tqdm == 4.50.0
+ scikit-learn == 0.24.1

## Dataset

| Dataset | Crystal | Property | Metric |
| :---: | :---: | :---: | :---: |
| Material Project | 69,239 | band_gap, formation_energy | MAE |
| DComPET | 74,939 | band_gap, total_energy, per_atom_energy, formation_energy, efermi, magnetization | MAE
| DComPET | 1,716 | band_gap (experimental) | MAE |
| DComPET | 54 | band_gap (experimental) | MAE |

The `Material Project` dataset can be referred to the [figshare](https://figshare.com/articles/dataset/Graphs_of_materials_project/7451351), their website could be found in [The materials project](https://www.materialsproject.org/), thanks a lot for their contribution in releasing the calculated crystal data.

## Preprocess
The key for preprocessing a new dataset with fitting our model is that generating `graph_cache.pickle` and `property.csv`.

The structure of `./data/material_project`:

You can download a zip files from the [google drive link](https://drive.google.com/drive/folders/1QlNO4Vs0Y28Zfnetzr9MJtakYJ_hwUn7?usp=sharing).

- `seed_0/`: a 9-fold cross-validation and independent test spliting example generated by `preprocess.py`.
    - `train_fold_[0-9].csv`: the training set for the corresponding fold number.
    - `valid_fold_[0-9].csv`: the validation set for the corresponding fold number.
    - `test.csv`: the independent test set.
- `atom_init.json`: the element fingerprint vector, with the format of `<atom_index>:<vector>`.
- `graph_cache.pickle`: the crystal graph dict, with the format of `<crystal_name>:<poscar_dict>`.
- `hubbard_u.yaml`: the atom radius dict.
- `preprocess.py`: the code of preprocessing data.
- `property.csv`: the table of crystal name with corresponding properties(band gap, total energy, per atom energy, formation energy, efermi, magnetization), a column with all 0's means without recording this property.
- `structures.tar.gz`: all crystal graph files with the format of `.poscar`.

The structure of `./data/DComPET`:

(2021.7.19 Notes: The dataset will be released later!)

Tips: The most time-consuming step is generating the `graph_cache.pickle` from the crystal graph files to the crystal graph dict object, thus we cache all dict object instead of generating it during training step.

## Training

For the band_gap property, run:

`nohup python -u train.py --gpu 0 --seed 0 --data_path ./data/material_project --train_path ./data/material_project --dataset_name band_gap --dataset_type regression --run_fold 1 --metric mae --save_dir ./ckpt/ensemble_band_gap --epochs 200 --init_lr 1e-4 --max_lr 3e-4 --final_lr 1e-4 --no_features_scaling --show_individual_scores > ./log/fold_1_band_gap.log 2>&1`

where the model will be stored at `./ckpt/ensemble_band_gap`, the training log will be stored at `./log`.

We also provide a bash script to run all training folds in parallel, please refer to `train_band_gap_*.sh`.

Some tips when training the model:
1. If you execute the code in the first time, it will generate `train_fold_{args.run_fold}_crystalnet.pickle`, `valid_fold_{args.run_fold}_crystalnet.pickle` and `test_crystalnet.pickle`, which will cost a few of time. And it will reload the pickle files when you executing the code for the second time, which help reduce the training time.

2. Hyperparameters could be found in the `./chemprop/parsing.py`, some key hyperparameters are listed below:
- `--radius`: The crystal neighbor radius, it will effect the number of neighbor atoms. If you revise this parameter, please regenerate the `.pickle` files in the step 1.
- `--rbf_parameters`: The key parameters for generating the Gaussian basis vectors. If you revise this parameter, please regenerate the `.pickle` files in the step 1.
- `--max_num_neighbors`: The maximum number of neighbors for each atoms. If you revise this parameter, please regenerate the `.pickle` files in the step 1.
- `--depth`: The number of stacking CMPNN blocks, unlike molecular graph, it will affect the final prediction result a lot for crystal graph since the over-smoothing phenomenon.

3. Although we load 6 properties for each crystal graph, we only use 1 of them to train the model during training, you can refer to `train.py`, line 72-88 to see how to fake the code to use only 1 property.

4. You may find the training time decrease after the first epoch, this is because we cached part of batch crystals in the memory, you can refer to `chemprop/features/featurization.py`, line 206-224, and revise the `len(CRYSTAL_TO_GRAPH) <= 10000`. The more batch crystals you cache, the less training time you cost and the more memories you use.

## Predict

For the band_gap property, run:
`nohup python -u predict.py --gpu 0 --seed 0 --data_path ./data/material_project --test_path ./data/material_project --dataset_name band_gap --checkpoint_dir ./ckpt/ensemble_band_gap/ --no_features_scaling > ./log/predict_band_gap.log 2>&1 &`

where the code will generate each fold of prediction in the `{args.test_path}/seed_{args.seed}/predict_{args.dataset_name}_fold_{fold_num}_crystalnet.csv` and ensemble all predictions for the final prediction in the `{args.test_path}/seed_{args.seed}/predict_{args.dataset_name}_crystalnet.csv`.

We also provide a bash script to run all training folds in parallel, please refer to `predict.sh`.

## Todo

- [ ] Clean the unuse function and write more comments.
- [ ] Add and clean the transfer learning code.
- [ ] Integrate other comparsion methods into this repository.
- [ ] Try our best to reduce the training time and the using memory, especially for the large dataset (long period).

## Website
You can come to our website for more related applications! 

[Matgen](https://matgen.nscc-gz.cn/)

## Citation

Please cite the following paper if you use this code in your work.
(Under developed...)