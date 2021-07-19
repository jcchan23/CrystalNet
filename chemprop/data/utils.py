from argparse import Namespace
import csv
from logging import Logger
import pickle
import random
from typing import List, Set, Tuple, Dict

from rdkit import Chem
import numpy as np
from tqdm import tqdm

from .data import CrystalDatapoint, CrystalDataset
from chemprop.features import load_features


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_task_names(path: str, use_compound_names: bool = False) -> Dict:
    """
    Gets the task names from a data CSV file.

    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A list of task names.
    """
    task_names = get_header(path)[1:]
    task_indices = {task_name: ind for ind, task_name in enumerate(task_names)}
    return task_indices


def get_num_tasks(path: str) -> int:
    """
    Gets the number of tasks in a data CSV file.

    :param path: Path to a CSV file.
    :return: The number of tasks.
    """
    return len(get_header(path)) - 1


def get_data(path: str,
             graph: dict = None,
             ari: object = None,
             gdf: object = None,
             radius_dic: object = None,
             skip_invalid_smiles: bool = False,
             args: Namespace = None,
             features_path: List[str] = None,
             max_data_size: int = None,
             use_compound_names: bool = None,
             logger: Logger = None) -> CrystalDataset:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    # :param path: Path to a CSV file.
    # :param graph: Path to a graph dict.
    # :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    # :param args: Arguments.
    # :param features_path: A list of paths to files containing features. If provided, it is used
    # in place of args.features_path.
    # :param max_data_size: The maximum number of data points to load.
    # :param use_compound_names: Whether file has compound names in addition to smiles strings.
    # :param logger: Logger.
    # :return: A CrystalDataset containing smiles strings and target values along
    # with other info such as additional features and compound names when desired.
    """
    debug = logger.debug if logger is not None else print

    # Prefer explicit function arguments but default to args if not provided
    if args is not None:
        features_path = features_path if features_path is not None else args.features_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        use_compound_names = use_compound_names if use_compound_names is not None else args.use_compound_names
    else:
        max_data_size, use_compound_names = float('inf'), False

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    # Load data
    crystal_dict = dict()
    with open(path) as f:
        reader = list(csv.reader(f))
        for line in reader[1:]:
            crystal_dict[line[0]] = [float(target) for target in line[1:]]

    # construct dataset
    data = CrystalDataset([
        CrystalDatapoint(
            crystal_name=name,
            crystal_dict=graph[name],
            targets=targets,
            ari=ari,
            gdf=gdf,
            radius_dic=radius_dic,
            args=args
        ) for name, targets in tqdm(crystal_dict.items(), total=len(crystal_dict))
    ], args=args)

    return data


def split_data(data: CrystalDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               args: Namespace = None,
               logger: Logger = None) -> Tuple[CrystalDataset, CrystalDataset, CrystalDataset]:
    """
    Splits data into training, validation, and test splits.

    :param data: A CrystalDataset.
    :param split_type: Split type.
    :param sizes: A length-3 tuple with the proportions of data in the
    train, validation, and test sets.
    :param seed: The random seed to use before shuffling data.
    :param args: Namespace of arguments.
    :param logger: A logger.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    if args is not None:
        folds_file, val_fold_index, test_fold_index = args.folds_file, args.val_fold_index, args.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None
    
    if split_type == 'crossval':
        print('=' * 45, 'crossval', '=' * 45)
        index_set = args.crossval_index_sets[args.seed]
        data_split = []
        for split in range(3):
            split_indices = index_set[split]
            data_split.append([data[i] for i in split_indices])
        train, val, test = tuple(data_split)
        print(f'train size: {len(train)}, val size: {len(val)}, test size: {len(test)}')
        return CrystalDataset(train), CrystalDataset(val), CrystalDataset(test)
    
    elif split_type == 'index_predetermined':
        split_indices = args.crossval_index_sets[args.seed]
        assert len(split_indices) == 3
        data_split = []
        for split in range(3):
            data_split.append([data[i] for i in split_indices[split]])
        train, val, test = tuple(data_split)
        return CrystalDataset(train), CrystalDataset(val), CrystalDataset(test)

    elif split_type == 'predetermined':
        if not val_fold_index:
            assert sizes[2] == 0  # test set is created separately so use all of the other data for train and val
        assert folds_file is not None
        assert test_fold_index is not None

        try:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f, encoding='latin1')  # in case we're loading indices from python2

        folds = [[data[i] for i in fold_indices] for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != test_fold_index and (val_fold_index is None or i != val_fold_index):
                train_val.extend(folds[i])

        if val_fold_index is not None:
            train = train_val
        else:
            random.seed(seed)
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        return CrystalDataset(train), CrystalDataset(val), CrystalDataset(test)

    elif split_type == 'random':
        data.shuffle(seed=seed)
        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))
        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]
        return CrystalDataset(train), CrystalDataset(val), CrystalDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')


def get_class_sizes(data: CrystalDataset) -> List[List[float]]:
    """
    Determines the proportions of the different classes in the classification dataset.

    :param data: A classification dataset
    :return: A list of lists of class proportions. Each inner list contains the class proportions
    for a task.
    """
    targets = data.targets()

    # Filter out Nones
    valid_targets = [[] for _ in range(data.num_tasks())]
    for i in range(len(targets)):
        for task_num in range(len(targets[i])):
            if targets[i][task_num] is not None:
                valid_targets[task_num].append(targets[i][task_num])

    class_sizes = []
    for task_targets in valid_targets:
        # Make sure we're dealing with a binary classification task
        assert set(np.unique(task_targets)) <= {0, 1}

        try:
            ones = np.count_nonzero(task_targets) / len(task_targets)
        except ZeroDivisionError:
            ones = float('nan')
            print('Warning: class has no targets')
        class_sizes.append([1 - ones, ones])

    return class_sizes


def validate_data(data_path: str) -> Set[str]:
    """
    Validates a data CSV file, returning a set of errors.

    :param data_path: Path to a data CSV file.
    :return: A set of error messages.
    """
    errors = set()

    header = get_header(data_path)

    with open(data_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        smiles, targets = [], []
        for line in reader:
            smiles.append(line[0])
            targets.append(line[1:])

    # Validate header
    if len(header) == 0:
        errors.add('Empty header')
    elif len(header) < 2:
        errors.add('Header must include task names.')

    mol = Chem.MolFromSmiles(header[0])
    if mol is not None:
        errors.add('First row is a SMILES string instead of a header.')

    # Validate smiles
    for smile in tqdm(smiles, total=len(smiles)):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            errors.add('Data includes an invalid SMILES.')

    # Validate targets
    num_tasks_set = set(len(mol_targets) for mol_targets in targets)
    if len(num_tasks_set) != 1:
        errors.add('Inconsistent number of tasks for each molecule.')

    if len(num_tasks_set) == 1:
        num_tasks = num_tasks_set.pop()
        if num_tasks != len(header) - 1:
            errors.add('Number of tasks for each molecule doesn\'t match number of tasks in header.')

    unique_targets = set(np.unique([target for mol_targets in targets for target in mol_targets]))

    if unique_targets <= {''}:
        errors.add('All targets are missing.')

    for target in unique_targets - {''}:
        try:
            float(target)
        except ValueError:
            errors.add('Found a target which is not a number.')

    return errors
