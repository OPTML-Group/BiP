import math
import os

import numpy as np
import torch
import torch.nn as nn

from models.layers import SubnetConvUnstructured, SubnetConvFilter, SubnetConvChannel, SubnetLinear


def freeze_vars(model, var_name, freeze_bn=False):
    """
    freeze vars. If freeze_bn then only freeze batch_norm params.
    """

    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if not isinstance(v, (nn.BatchNorm2d, nn.BatchNorm2d)) or freeze_bn:
                if getattr(v, var_name) is not None:
                    getattr(v, var_name).requires_grad = False


def unfreeze_vars(model, var_name):
    assert var_name in ["weight", "bias", "popup_scores"]
    for i, v in model.named_modules():
        if hasattr(v, var_name):
            if getattr(v, var_name) is not None:
                getattr(v, var_name).requires_grad = True


def set_prune_rate_model(model, prune_rate):
    for _, v in model.named_modules():
        if hasattr(v, "set_prune_rate"):
            v.set_prune_rate(prune_rate)


def set_prune_rate_model_smart_ratio(model, prune_rate):
    print("Prune the model at the rate of {}".format(prune_rate))
    keep_count_list = []
    pram_count_list = []
    j = 0
    for i, (n, v) in enumerate(model.named_modules()):

        if hasattr(v, "set_prune_rate"):
            keep_count = ((21 - j) ** 2 + (21 - j)) * math.prod(v.popup_scores.shape)
            j += 1
            keep_count_list.append(keep_count)
            pram_count_list.append(math.prod(v.popup_scores.shape))
    total_count = sum(pram_count_list)

    keep_count = total_count * prune_rate
    rest_count = keep_count - pram_count_list[-1] * 0.3

    share = rest_count / sum(keep_count_list[0:-1])

    ratio_list = []
    j = 0
    for i, (n, v) in enumerate(model.named_modules()):
        if hasattr(v, "set_prune_rate"):
            keep_ratio = ((21 - j) ** 2 + (21 - j)) * share
            j += 1
            ratio_list.append(keep_ratio)
    ratio_list[-1] = 0.3

    param_count_kept_list = []
    j = 0
    for i, (n, v) in enumerate(model.named_modules()):

        if hasattr(v, "set_prune_rate"):
            param_count_kept = ratio_list[j] * pram_count_list[j]
            param_count_kept_list.append(param_count_kept)
            j += 1

    j = 0
    for i, (n, v) in enumerate(model.named_modules()):

        if hasattr(v, "set_prune_rate"):
            v.set_prune_rate(ratio_list[j])
            j += 1


def get_layers(layer_type):
    """
        Returns: (conv_layer, linear_layer)
    """
    if layer_type == "dense":
        return nn.Conv2d, nn.Linear
    elif layer_type == "unstructured":
        # Unstructured pruning
        return SubnetConvUnstructured, SubnetLinear
    elif layer_type == "channel":
        # Structured channel-wise pruning
        return SubnetConvChannel, nn.Linear
    elif layer_type == "filter":
        # Structured filter-wise pruning
        return SubnetConvFilter, nn.Linear
    else:
        raise ValueError("Incorrect layer type")


def show_gradients(model):
    for i, v in model.named_parameters():
        print(f"variable = {i}, Gradient requires_grad = {v.requires_grad}")
        pass


def get_score_gradient_function(model, score_gradient):
    grad_scalar_list = []
    ind = 0
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                getattr(v, "popup_scores").grad.retain_graph = True
                grad_scalar_list.append(torch.sum(getattr(v, "popup_scores").grad * score_gradient[ind]))
                ind += 1
    grad_scalar = torch.tensor(sum(grad_scalar_list), requires_grad=True)
    return grad_scalar


def get_score_gradient(model):
    grad_list = []
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                grad_list.append(getattr(v, "popup_scores").grad.detach())
    return grad_list


def get_scores(model):
    grad_list = []
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                grad_list.append(getattr(v, "popup_scores"))
    return grad_list


def get_param(model):
    grad_list = []
    for i, v in model.named_modules():
        if hasattr(v, "weight"):
            if getattr(v, "weight") is not None:
                grad_list.append(getattr(v, "weight"))
        if hasattr(v, "bias"):
            if getattr(v, "bias") is not None:
                grad_list.append(getattr(v, "bias"))
    return grad_list


def initialize_scores(model, init_type):
    print(f"Initialization relevance score with {init_type} initialization")
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            if init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.popup_scores)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.popup_scores)
            elif init_type == "xavier_uniform":
                nn.init.xavier_uniform_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(
                    m.popup_scores, gain=nn.init.calculate_gain("relu")
                )


def initialize_scaled_score(model):
    print(
        "Initialization relevance score proportional to weight magnitudes (OVERWRITING SOURCE NET SCORES)"
    )
    for m in model.modules():
        if hasattr(m, "popup_scores"):
            m.popup_scores.data = m.weight.data


def scale_rand_init(model, k):
    print(
        f"Initializating random weight with scaling by 1/sqrt({k}) | Only applied to CONV & FC layers"
    )
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.weight.data = 1 / math.sqrt(k) * m.weight.data


def switch_to_prune(model):
    # print(f"#################### Pruning network ####################")
    # print(f"===>>  gradient for weights: None  | training importance scores only")

    unfreeze_vars(model, "popup_scores")
    freeze_vars(model, "weight")
    freeze_vars(model, "bias")


def switch_to_finetune(model):
    # print(f"#################### Fine-tuning network ####################")
    # print(
    #     f"===>>  gradient for importance_scores: None  | fine-tuning important weigths only"
    # )
    freeze_vars(model, "popup_scores")
    unfreeze_vars(model, "weight")
    unfreeze_vars(model, "bias")


def switch_to_bilevel(model):
    unfreeze_vars(model, "popup_scores")
    unfreeze_vars(model, "weight")
    unfreeze_vars(model, "bias")


def prepare_model(model, args):
    """
        1. Set model pruning rate
        2. Set gradients base on training mode.
    """

    set_prune_rate_model(model, args.k)

    if args.exp_mode == "pretrain":
        print(f"#################### Pre-training network ####################")
        print(f"===>>  gradient for importance_scores: None | training weights only")
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")

    elif args.exp_mode == "prune":
        print(f"#################### Pruning network ####################")
        print(f"===>>  gradient for weights: None | training importance scores only")

        unfreeze_vars(model, "popup_scores")
        freeze_vars(model, "weight", args.freeze_bn)
        freeze_vars(model, "bias", args.freeze_bn)

    elif args.exp_mode == "finetune":
        print(f"#################### Fine-tuning network ####################")
        print(f"===>>  gradient for importance_scores: None | fine-tuning important weights only")
        freeze_vars(model, "popup_scores", args.freeze_bn)
        unfreeze_vars(model, "weight")
        unfreeze_vars(model, "bias")

    else:
        assert False, f"{args.exp_mode} mode is not supported"

    initialize_scores(model, args.scores_init_type)


def subnet_to_dense(subnet_dict, p):
    """
        Convert a subnet state dict (with subnet layers) to dense i.e., which can be directly 
        loaded in network with dense layers.
    """
    dense = {}

    # load dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" not in k:
            dense[k] = v

    # update dense variables
    for (k, v) in subnet_dict.items():
        if "popup_scores" in k:
            s = torch.abs(subnet_dict[k])

            out = s.clone()
            _, idx = s.flatten().sort()
            j = int((1 - p) * s.numel())

            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1
            dense[k.replace("popup_scores", "weight")] = (
                    subnet_dict[k.replace("popup_scores", "weight")] * out
            )
    return dense


def dense_to_subnet(model, state_dict):
    """
        Load a dict with dense-layer in a model trained with subnet layers. 
    """
    model.load_state_dict(state_dict, strict=False)


def current_model_pruned_fraction(model, result_dir, verbose=True):
    """
        Find pruning raio per layer. Return average of them.
        Result_dict should correspond to the checkpoint of model.

        DEV: This actually works for smart ratio, although it may suggest
        layers are pruned evenly with smart ratio.
    """

    # load the dense models
    path = os.path.join(result_dir, "checkpoint_dense.pth.tar")

    pl = []

    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name + ".weight" in state_dict.keys():
                    d = state_dict[name + ".weight"].data.cpu().numpy()
                    p = 100 * np.sum(d == 0) / np.size(d)
                    pl.append(p)
                    if verbose:
                        print(name, module, p)
        return np.mean(pl)


def sanity_check_paramter_updates(model, last_ckpt):
    """
        Check whether weigths/popup_scores gets updated or not compared to last ckpt.
        ONLY does it for 1 layer (to avoid computational overhead)
    """
    for i, v in model.named_modules():
        if hasattr(v, "weight") and hasattr(v, "popup_scores"):
            if getattr(v, "weight") is not None:
                w1 = getattr(v, "weight").data.cpu()
                w2 = last_ckpt[i + ".weight"].data.cpu()
            if getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                s2 = last_ckpt[i + ".popup_scores"].data.cpu()
            return not torch.allclose(w1, w2), not torch.allclose(s1, s2)


def extract_mask_as_tensor(model, k):
    score_list = []
    for (name, vec) in model.named_modules():
        if hasattr(vec, "popup_scores"):
            attr = getattr(vec, "popup_scores")
            if attr is not None:
                score_list.append(attr.view(-1))
    scores = torch.cat(score_list)
    mask = scores.clone()
    _, idx = scores.flatten().sort()
    j = int((1 - k) * scores.numel())

    flat_out = mask.flatten()
    flat_out[idx[:j]] = 0
    flat_out[idx[j:]] = 1

    return mask


def calculate_IOU(mask1, mask2):
    mask1 = mask1.view(-1)
    mask2 = mask2.view(-1)
    assert mask1.shape[0] == mask2.shape[0]
    intersection = ((mask1 == 1) & (mask2 == 1)).sum(0)
    union = ((mask1 == 1) | (mask2 == 1)).sum(0)
    iou_score = intersection / union
    return iou_score
