from argparse import Namespace
import logging
from typing import Callable, List, Union

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange

from chemprop.data import CrystalDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: nn.Module,
          data: Union[CrystalDataset, List[CrystalDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A CrystalDataset (or a list of CrystalDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print
    
    model.train()
    
    data.shuffle()

    loss_sum, iter_count = 0, 0

    num_iters = len(data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability

    for i in trange(0, num_iters, args.batch_size):
        # Prepare batch
        if i + args.batch_size > len(data):
            break
        crystal_batch = CrystalDataset(data[i:i + args.batch_size])

        mask = torch.Tensor([[x is not None for x in tb] for tb in crystal_batch.targets()])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in crystal_batch.targets()])
        class_weights = torch.ones(targets.shape)

        if args.cuda:
            mask, targets, class_weights = mask.cuda(), targets.cuda(), class_weights.cuda()

        # Run model
        model.zero_grad()
        preds_batch = model(crystal_batch)  # (batch, multiclass)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds_batch[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds_batch.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds_batch, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()
        
        loss_sum += loss.item()
        iter_count += len(crystal_batch)

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(crystal_batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for j, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{j}', lr, n_iter)

    return n_iter


def transfer_train(model: nn.Module,
          data: Union[CrystalDataset, List[CrystalDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:

    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A CrystalDataset (or a list of CrystalDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()

    data.shuffle()

    loss_sum, iter_count = 0, 0

    num_iters = len(data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability

    for i in trange(0, num_iters, args.batch_size):
        # Prepare batch
        if i + args.batch_size > len(data):
            break
        crystal_batch = CrystalDataset(data[i:i + args.batch_size])
        mask = torch.Tensor([[x is not None for x in tb] for tb in crystal_batch.targets()])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in crystal_batch.targets()])
        class_weights = torch.ones(targets.shape)

        if args.cuda:
            mask, targets, class_weights = mask.cuda(), targets.cuda(), class_weights.cuda()

        # Run model
        model.zero_grad()
        preds_batch = model(crystal_batch)  # (batch, multiclass)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat(
                [loss_func(preds_batch[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in
                 range(preds_batch.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds_batch, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += len(crystal_batch)

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(crystal_batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for j, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{j}', lr, n_iter)

    return n_iter


