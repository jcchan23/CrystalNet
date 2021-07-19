from typing import List
import torch
import torch.nn as nn

from chemprop.data import CrystalDataset, StandardScaler


def predict(model: nn.Module,
            data: CrystalDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A CrystalDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        # Prepare batch
        crystal_batch = CrystalDataset(data[i:i + batch_size])

        with torch.no_grad():
            preds_batch = model(crystal_batch)

        preds_batch = preds_batch.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            preds_batch = scaler.inverse_transform(preds_batch)

        # Collect vectors
        preds_batch = preds_batch.tolist()
        preds.extend(preds_batch)

    return preds


def transfer_predict(model: nn.Module,
                     data: CrystalDataset,
                     batch_size: int,
                     scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A CrystalDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        # Prepare batch
        crystal_batch = CrystalDataset(data[i:i + batch_size])

        with torch.no_grad():
            preds_batch = model(crystal_batch)

        preds_batch = preds_batch.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            preds_batch = scaler.inverse_transform(preds_batch)

        # Collect vectors, None for keeping dimension
        preds.extend(preds_batch.tolist())

    return preds

