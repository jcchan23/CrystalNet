from argparse import Namespace
from logging import Logger
import os
import pickle
from typing import Tuple
import numpy as np
import torch

from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names, get_data
from chemprop.utils import makedirs
from chemprop.parsing import parse_train_args
from chemprop.utils import create_logger
from chemprop.features import AtomCustomJSONInitializer, GaussianDistance, load_radius_dict
from chemprop.data import CrystalDataset


def run(args: Namespace, logger: Logger = None) -> Tuple[np.ndarray, np.ndarray]:
    info = logger.info if logger is not None else print
    info('=' * 20 + f' training on fold {args.run_fold} ' + '=' * 20)

    # Load feature object
    ari = AtomCustomJSONInitializer(f'{args.data_path}/atom_init.json')
    dmin, dmax, step, var = args.rbf_parameters
    gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step, var=var)
    radius_dic = load_radius_dict(f'{args.data_path}/hubbard_u.yaml')

    # Load and cache data
    info('Loading data')
    if os.path.exists(f'{args.train_path}/graph_cache.pickle'):
        with open(f'{args.train_path}/graph_cache.pickle', 'rb') as f:
            all_graph = pickle.load(f)
    elif os.path.exists(f'{args.data_path}/graph_cache.pickle'):
        with open(f'{args.data_path}/graph_cache.pickle', 'rb') as f:
            all_graph = pickle.load(f)
    else:
        assert "There is no poscar graph cache, please use preprocess.py to generate poscar graph cache!"

    if os.path.exists(f'{args.train_path}/seed_{args.seed}/test_crystalnet.pickle'):
        with open(f'{args.train_path}/seed_{args.seed}/test_crystalnet.pickle', 'rb') as f:
            test_data = pickle.load(f)
    else:
        test_data = get_data(path=f'{args.train_path}/seed_{args.seed}/test.csv',
                             graph=all_graph, ari=ari, gdf=gdf, radius_dic=radius_dic, args=args, logger=logger)
        with open(f'{args.train_path}/seed_{args.seed}/test_crystalnet.pickle', 'wb') as fw:
            pickle.dump(test_data, fw)

    # assert False
    if os.path.exists(f'{args.train_path}/seed_{args.seed}/train_fold_{args.run_fold}_crystalnet.pickle'):
        with open(f'{args.train_path}/seed_{args.seed}/train_fold_{args.run_fold}_crystalnet.pickle', 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = get_data(path=f'{args.train_path}/seed_{args.seed}/train_fold_{args.run_fold}.csv',
                              graph=all_graph, ari=ari, gdf=gdf, radius_dic=radius_dic, args=args, logger=logger)
        with open(f'{args.train_path}/seed_{args.seed}/train_fold_{args.run_fold}_crystalnet.pickle', 'wb') as fw:
            pickle.dump(train_data, fw)

    if os.path.exists(f'{args.train_path}/seed_{args.seed}/valid_fold_{args.run_fold}_crystalnet.pickle'):
        with open(f'{args.train_path}/seed_{args.seed}/valid_fold_{args.run_fold}_crystalnet.pickle', 'rb') as f:
            valid_data = pickle.load(f)
    else:
        valid_data = get_data(path=f'{args.train_path}/seed_{args.seed}/valid_fold_{args.run_fold}.csv',
                              graph=all_graph, ari=ari, gdf=gdf, radius_dic=radius_dic, args=args, logger=logger)
        with open(f'{args.train_path}/seed_{args.seed}/valid_fold_{args.run_fold}_crystalnet.pickle', 'wb') as fw:
            pickle.dump(valid_data, fw)

    # subsample for incremental experiment
    if args.max_data_size != float('inf'):
        train_data.shuffle(seed=args.seed)
        train_data = CrystalDataset(train_data[:args.max_data_size], args=args)

    task_indices = get_task_names(path=f'{args.train_path}/property.csv', use_compound_names=True)
    args.task_index = task_indices[args.dataset_name]
    args.task_names = [args.dataset_name]
    args.num_tasks = 1
    info(task_indices)
    info(args.task_names)
    info(args.task_index)

    # convert multi targets to single target, just temp using before revising to multitask
    train_targets = [[targets[args.task_index]] for targets in train_data.targets()]
    train_data.set_targets(train_targets)

    valid_targets = [[targets[args.task_index]] for targets in valid_data.targets()]
    valid_data.set_targets(valid_targets)

    test_targets = [[targets[args.task_index]] for targets in test_data.targets()]
    test_data.set_targets(test_targets)

    info(f'Total size = {len(train_data)+len(valid_data)+len(test_data):,} | '
         f'train size = {len(train_data):,}({len(train_data)/(len(train_data)+len(valid_data)+len(test_data)):.1f}) | '
         f'valid size = {len(valid_data):,}({len(valid_data)/(len(train_data)+len(valid_data)+len(test_data)):.1f}) | '
         f'test size = {len(test_data):,}({len(test_data)/(len(train_data)+len(valid_data)+len(test_data)):.1f})')

    # Required for NormLR
    args.train_data_size = len(train_data)

    # Training
    save_dir = os.path.join(args.save_dir, f'fold_{args.run_fold}')
    makedirs(save_dir)
    valid_scores, test_scores = run_training(train_data, valid_data, test_data, args, logger)

    # Report scores
    for task_name, valid_score, test_score in zip(args.task_names, valid_scores, test_scores):
        info(f'Task name "{task_name}": Valid {args.metric} = {valid_score:.6f}, Test {args.metric} = {test_score:.6f}')

    return np.nanmean(valid_scores), np.nanmean(test_scores)


if __name__ == '__main__':
    args = parse_train_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    mean_valid_score, mean_test_score = run(args, logger)
    print(f'Results on the fold {args.run_fold}')
    print(f'Overall Valid {args.metric}: {mean_valid_score:.5f}, Test scores: {mean_test_score:.5f}')
