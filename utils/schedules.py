import sys

import numpy as np
import torch


def get_k_at_epoch(epoch, args, method="Linear", IMP_epoch=10):
    if method == "Linear":
        k = 1 - (epoch + 1) / args.epochs * (1 - args.k)
    elif method == "IMP-style-Linear":
        k = 1 - ((epoch + 1) // IMP_epoch + 1) / (args.epochs // IMP_epoch) * (1 - args.k)
    elif method == "IMP-style-Exponential":
        max_stages = args.epochs // IMP_epoch
        epoch_stage = ((epoch + 1) // IMP_epoch + 1) / max_stages
        if epoch_stage > 1:
            epoch_stage = 1
        p = args.k
        k = np.power(p, epoch_stage)  # e.g., 4= np.power(8,2/3)

    return k


def get_lr_policy(lr_schedule):
    """Implement a new schduler directly in this file. 
    Args should contain a single choice for learning rate scheduler."""

    d = {
        "constant": constant_schedule,
        "cosine": cosine_schedule,
        "step": step_schedule,
    }
    return d[lr_schedule]


def get_mask_lr_policy(lr_schedule):
    """Implement a new schduler directly in this file.
    Args should contain a single choice for learning rate scheduler."""

    d = {
        "cosine": cosine_mask_schedule,
        "step": step_mask_schedule,
    }
    return d[lr_schedule]


def get_optimizer(model, args):
    if args.optimizer == "sgd":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif args.optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, )
    elif args.optimizer == "rmsprop":
        optim = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    else:
        print(f"{args.optimizer} is not supported.")
        sys.exit(0)
    return optim


def set_new_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def constant_schedule(optimizer, args):
    def set_lr(epoch, lr=args.lr, epochs=args.epochs):
        if epoch < args.warmup_epochs:
            lr = args.warmup_lr

        set_new_lr(optimizer, lr)

    return set_lr


def cosine_schedule(optimizer, args):
    def set_lr(epoch, lr=args.lr, epochs=args.epochs):
        if epoch < args.warmup_epochs:
            a = args.warmup_lr
        else:
            epoch = epoch - args.warmup_epochs
            a = lr * 0.5 * (1 + np.cos((epoch - 1) / epochs * np.pi))

        set_new_lr(optimizer, a)

    return set_lr


def step_schedule(optimizer, args):
    def set_lr(epoch, lr=args.lr, epochs=args.epochs):
        if epoch < args.warmup_epochs:
            a = args.warmup_lr
        else:
            epoch = epoch - args.warmup_epochs

        a = lr
        if epoch >= 0.5 * epochs:
            a = lr * 0.1
        if epoch >= 0.75 * epochs:
            a = lr * 0.01

        set_new_lr(optimizer, a)

    return set_lr


def cosine_mask_schedule(optimizer, args):
    def set_lr(epoch, lr=args.mask_lr, epochs=args.epochs):
        if epoch < args.warmup_epochs:
            a = args.warmup_lr
        else:
            epoch = epoch - args.warmup_epochs
            a = lr * 0.5 * (1 + np.cos((epoch - 1) / epochs * np.pi))

        set_new_lr(optimizer, a)

    return set_lr


def step_mask_schedule(optimizer, args):
    def set_lr(epoch, lr=args.mask_lr, epochs=args.epochs):
        if epoch < args.warmup_epochs:
            a = args.warmup_lr
        else:
            epoch = epoch - args.warmup_epochs

        a = lr
        if epoch >= 0.75 * epochs:
            a = lr * 0.1
        if epoch >= 0.9 * epochs:
            a = lr * 0.01
        if epoch >= epochs:
            a = lr * 0.001

        set_new_lr(optimizer, a)

    return set_lr
