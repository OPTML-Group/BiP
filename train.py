from __future__ import absolute_import
from __future__ import print_function

import importlib
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

import datasets
import models
from args import parse_args
from utils.general_utils import (
    save_checkpoint,
    create_subdirs,
    parse_configs_file,
    clone_results_to_latest_subdir,
    setup_seed
)
from utils.model import extract_mask_as_tensor, calculate_IOU
from utils.model import (
    get_layers,
    prepare_model,
    initialize_scaled_score,
    scale_rand_init,
    current_model_pruned_fraction,
)
from utils.schedules import get_lr_policy, get_optimizer


def main():
    args = parse_args()
    if args.configs is not None:
        parse_configs_file(args)

    # sanity checks
    if args.exp_mode in ["prune", "finetune"] and not args.resume:
        assert args.source_net, "Provide checkpoint to prune/finetune"

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(Path(args.result_dir), args.exp_name, args.exp_mode)

    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
    else:
        n = 0
    os.makedirs(result_main_dir, exist_ok=True)
    result_sub_dir = os.path.join(
        result_main_dir,
        "{}--k-{:.4f}_trainer-{}_epochs-{}_arch-{}".format(
            n,
            args.k,
            args.trainer,
            args.epochs,
            args.arch,
        ),
    )
    create_subdirs(result_sub_dir)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    setup_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create model
    # ConvLayer and LinearLayer are classes, not instances.
    ConvLayer, LinearLayer = get_layers(args.layer_type)
    unstructured = True if args.layer_type == "unstructured" else False
    model = models.__dict__[args.arch](
        ConvLayer, LinearLayer, num_classes=args.num_classes,
        k=args.k, unstructured=unstructured
    ).to(device)

    # Customize models for training/pruning/fine-tuning
    prepare_model(model, args)

    # Dataloader
    D = datasets.__dict__[args.dataset](args, normalize=args.normalize)
    train_loader, val_loader, test_loader = D.data_loaders()

    # autograd
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)

    # For bi-level only
    mask_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.mask_lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

    mask_lr_policy = get_lr_policy(args.mask_lr_schedule)(mask_optimizer, args)

    # train & val method
    trainer = importlib.import_module(f"trainer.{args.trainer}").train
    val = getattr(importlib.import_module("utils.eval"), args.val_method)

    # Load source_net (if checkpoint provided). 
    # Only load the state_dict (required for pruning and fine-tuning)
    if args.source_net:
        if os.path.isfile(args.source_net):
            logger.info("=> loading source model from '{}'".format(args.source_net))
            checkpoint = torch.load(args.source_net, map_location=device)
            if args.source_net.split(".")[-1] == "pt":
                checkpoint = {"state_dict": checkpoint}
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            logger.info("=> loaded checkpoint '{}'".format(args.source_net))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.source_net))

    # Init scores once source net is loaded.
    if args.exp_mode == "prune":
        if args.scaled_score_init:
            # NOTE: scaled_init_scores will overwrite the scores in the pre-trained net.
            initialize_scaled_score(model)
        else:
            # Scaled random initialization. Useful when training a high sparse net from scratch.
            # If not used, a sparse net (without batch-norm) from scratch will not converge.
            # With batch-norm its not really necessary.
            scale_rand_init(model, args.k)

    best_prec1 = 0
    start_epoch = 0
    assert not (args.source_net and args.resume), (
        "Incorrect setup: "
        "resume => required to resume a previous experiment (loads all parameters)|| "
        "source_net => required to start pruning/fine-tuning from a source model (only load state_dict)"
    )
    # resume (if checkpoint provided). Continue training with previous settings.
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    # Evaluate
    if args.evaluate or args.exp_mode in ["finetune"]:
        p1, _ = val(model, device, test_loader, criterion, args, None)
        logger.info(f"Validation accuracy {args.val_method} for source-net: {p1}")
        if args.evaluate:
            return

    iou_res = []
    last_mask = extract_mask_as_tensor(model, args.k)

    # Start training
    for epoch in range(start_epoch, args.epochs + args.warmup_epochs):
        start = time.time()
        lr_policy(epoch)
        mask_lr_policy(epoch)

        if args.trainer == "bilevel":
            optimizer = (optimizer, mask_optimizer)

        # train
        trainer(
            model,
            device,
            (train_loader, val_loader),
            criterion,
            optimizer,
            epoch,
            args,
        )

        current_mask = extract_mask_as_tensor(model, args.k)
        cur_iou = calculate_IOU(last_mask, current_mask)
        iou_res.append(cur_iou)
        last_mask = current_mask

        logger.info(f"Current IoU: {cur_iou}")

        # evaluate on test set
        if args.val_method == "smooth":
            prec1, radii = val(
                model, device, test_loader, criterion, args, epoch
            )
            logger.info(f"Epoch {epoch}, mean provable Radii  {radii}")
        prec1, _ = val(model, device, test_loader, criterion, args, epoch)

        # remember best prec@1 and save checkpoint
        if args.trainer == "bilevel":
            optimizer = optimizer[0]
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args,
            result_dir=os.path.join(result_sub_dir, "checkpoint"),
            save_dense=args.save_dense,
        )

        clone_results_to_latest_subdir(
            result_sub_dir, os.path.join(result_main_dir, "latest_exp")
        )

        logger.info("This epoch duration :{}".format(time.time() - start))

        logger.info(
            f"Epoch {epoch}, val-method {args.val_method}, validation accuracy {prec1}, best_prec {best_prec1}"
        )

    save_checkpoint(
        {
            "epoch": args.epochs,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_prec1": best_prec1,
            "optimizer": optimizer.state_dict(),
        },
        True if args.epochs == 0 else False,
        args,
        result_dir=os.path.join(result_sub_dir, "checkpoint"),
        save_dense=args.save_dense,
    )

    clone_results_to_latest_subdir(
        result_sub_dir, os.path.join(result_main_dir, "latest_exp")
    )

    current_model_pruned_fraction(
        model, os.path.join(result_sub_dir, "checkpoint"), verbose=True
    )


if __name__ == "__main__":
    main()
