# small-caps refers to cifar-style models i.e., resnet18 -> for cifar vs ResNet18 -> standard arch.
from models.vgg_cifar import (
    vgg2,
    vgg2_bn,
    vgg4,
    vgg4_bn,
    vgg6,
    vgg6_bn,
    vgg8,
    vgg8_bn,
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
)
from models.resnet_cifar import resnet18, resnet34, resnet50, resnet101, resnet152
from models.wrn_cifar import wrn_28_10, wrn_28_1, wrn_28_4, wrn_34_10, wrn_40_2
from models.basic import (
    lin_1,
    lin_2,
    lin_3,
    lin_4,
    mnist_model,
    mnist_model_large,
    cifar_model,
    cifar_model_large,
    cifar_model_resnet,
    vgg4_without_maxpool,
)

from models.resnet import ResNet18, ResNet34, ResNet50
from models.resnets import resnet20s, resnet32s, resnet44s, resnet56s

__all__ = [
    "vgg2",
    "vgg2_bn",
    "vgg4",
    "vgg4_bn",
    "vgg6",
    "vgg6_bn",
    "vgg8",
    "vgg8_bn",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "wrn_28_10",
    "wrn_28_1",
    "wrn_28_4",
    "wrn_34_10",
    "wrn_40_2",
    "lin_1",
    "lin_2",
    "lin_3",
    "lin_4",
    "mnist_model",
    "mnist_model_large",
    "cifar_model",
    "cifar_model_large",
    "cifar_model_resnet",
    "vgg4_without_maxpool",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "resnet20s"
    "resnet56s"
]
