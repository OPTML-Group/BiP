import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # primary
    parser.add_argument(
        "--configs", type=str, default=None, help="configs file",
    )
    parser.add_argument(
        "--result-dir",
        default="./trained_models",
        type=str,
        help="directory to save results",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Name of the experiment (creates dir with this name in --result-dir)",
    )

    parser.add_argument(
        "--exp-mode",
        type=str,
        choices=("pretrain", "prune", "finetune"),
        help="Train networks following one of these methods.",
    )

    # Model
    parser.add_argument("--arch", type=str, help="Model achitecture")
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes in the model",
    )
    parser.add_argument(
        "--layer-type", type=str, choices=("dense", "unstructured", "channel", "filter"), help="dense | unstructured | channel | filter"
    )

    # Pruning
    parser.add_argument(
        "--k",
        type=float,
        default=1.0,
        help="Fraction of weight variables kept in subnet",
    )

    parser.add_argument(
        "--scaled-score-init",
        action="store_true",
        default=False,
        help="Init importance scores proportaional to weights (default kaiming init)",
    )

    parser.add_argument(
        "--scale-rand-init",
        action="store_true",
        default=False,
        help="Init weight with scaling using pruning ratio",
    )

    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        default=False,
        help="freeze batch-norm parameters in pruning",
    )

    parser.add_argument(
        "--source-net",
        type=str,
        default=None,
        help="Checkpoint which will be pruned/fine-tuned",
    )

    parser.add_argument(
        "--scores-init-type",
        choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal"),
        help="Which init to use for relevance scores",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["CIFAR10", "CIFAR100", "TinyImageNet", "ImageNet", "ImageNetOrigin", "ImageNetLMDB"],
        help="Dataset for training and eval",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        metavar="N",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="whether to normalize the data",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="path to datasets"
    )

    parser.add_argument(
        "--image-dim", type=int, default=32, help="Image size: dim x dim x 3"
    )

    # Training
    parser.add_argument(
        "--trainer",
        type=str,
        default="base",
        choices=["bilevel", "bilevel_finetune", "base"],
        help="Natural (base) or adversarial or verifiable training",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, metavar="N", help="number of epochs to train"
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=("sgd", "adam", "rmsprop")
    )
    parser.add_argument("--wd", default=5e-4, type=float, help="Weight decay")

    parser.add_argument("--mask-lr", type=float, default=0.1, help="mask learning rate for bi-level only")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--mask-lr-schedule",
        type=str,
        default="cosine",
        choices=("cosine", "step"),
        help="lr scheduler for finetuning in bi-level problem"
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=("step", "cosine"),
        help="Learning rate schedule",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument(
        "--warmup-epochs", type=int, default=0, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--warmup-lr", type=float, default=0.1, help="warmup learning rate"
    )
    parser.add_argument(
        "--save-dense",
        action="store_true",
        default=False,
        help="Save dense model alongwith subnets.",
    )

    # Evaluate
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate model"
    )

    parser.add_argument(
        "--val-method",
        type=str,
        default="base",
        choices=["base"],
        help="base: evaluation on unmodified inputs",
    )

    # Restart
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="path to latest checkpoint (default:None)",
    )

    # Additional
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--print-freq",
        type=int,
        default=10,
        help="Number of batches to wait before printing training logs",
    )

    parser.add_argument(
        "--lr2",
        type=float,
        default=0.1,
        help="learning rate for the second term",
    )

    parser.add_argument(
        "--accelerate",
        action="store_true",
        help="Use PFTT to accelerate",
    )

    return parser.parse_args()
