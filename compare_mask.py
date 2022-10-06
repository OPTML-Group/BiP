from __future__ import absolute_import
from __future__ import print_function
import argparse

import models
from utils.model import *

parser = argparse.ArgumentParser(description="Mask Comparison")
parser.add_argument("--arch", default="resnet18")
parser.add_argument("--model_path1", type=str)
parser.add_argument("--model_path2", type=str)
parser.add_argument("--k", type=float, default=0.0005)
parser.add_argument("--num_classes", type=int, default=10)

args = parser.parse_args()

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create model
    # ConvLayer and LinearLayer are classes, not instances.
    ConvLayer, LinearLayer = get_layers("subnet")
    model_1 = models.__dict__[args.arch](
        ConvLayer, LinearLayer, "kaiming_normal", num_classes=args.num_classes, k=args.k
    ).to(device)

    model_2 = models.__dict__[args.arch](
        ConvLayer, LinearLayer, "kaiming_normal", num_classes=args.num_classes, k=args.k
    ).to(device)

    checkpoint_1 = torch.load(args.model_path1, map_location=device)
    model_1.load_state_dict(checkpoint_1["state_dict"])

    checkpoint_2 = torch.load(args.model_path2, map_location=device)
    model_2.load_state_dict(checkpoint_2["state_dict"])

    mask1 = extract_mask_as_tensor(model_1, args.k)
    mask2 = extract_mask_as_tensor(model_2, args.k)

    iou_score = calculate_IOU(mask1, mask2)

    print(f"The IOU between the two masks is {iou_score}")
