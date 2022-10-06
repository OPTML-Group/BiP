import os
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import Lambda
from datasets.lmdb_dataset import ImageFolderLMDB


# NOTE: Each dataset class must have public norm_layer, tr_train, tr_test objects.
# These are needed for ood/semi-supervised dataset used alongwith in the training and eval.
class imagenet:
    """
        imagenet dataset.
    """

    def __init__(self, args, normalize=True):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.tr_train = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = datasets.ImageFolder(
            os.path.join(self.args.data_dir, "train"), self.tr_train
        )
        testset = datasets.ImageFolder(
            os.path.join(self.args.data_dir, "val"), self.tr_test
        )

        np.random.seed(10)

        train_loader = DataLoader(
            trainset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=True,
            **kwargs,
        )
        np.random.seed(50)
        val_loader = DataLoader(
            trainset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=True,
            **kwargs,
        )
        test_loader = DataLoader(
            testset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            **kwargs,
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader


class imagenet_lmbd:
    """
            imagenet dataset.
        """

    def __init__(self, args, normalize=True):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.tr_train = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        self.tr_test = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = ImageFolderLMDB(
            os.path.join(self.args.data_dir, "train.lmdb"), self.tr_train
        )
        testset = ImageFolderLMDB(
            os.path.join(self.args.data_dir, "val.lmdb"), self.tr_test
        )

        np.random.seed(10)

        train_loader = DataLoader(
            trainset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=True,
            **kwargs,
        )
        np.random.seed(50)
        val_loader = DataLoader(
            trainset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=True,
            **kwargs,
        )
        test_loader = DataLoader(
            testset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            **kwargs,
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader


class imagenet_h5py:
    """
        imagenet dataset.
    """

    def __init__(self, args, normalize=True):
        self.args = args

        self.norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.tr_train = [
            Lambda(lambda x: x / 255.),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
        ]
        self.tr_test = [
            Lambda(lambda x: x / 255.),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
        ]

        if normalize:
            self.tr_train.append(self.norm_layer)
            self.tr_test.append(self.norm_layer)

        self.tr_train = transforms.Compose(self.tr_train)
        self.tr_test = transforms.Compose(self.tr_test)

    def data_loaders(self, **kwargs):
        trainset = IMAGENET(root=self.args.data_dir, split='train', transform=self.tr_train)
        testset = IMAGENET(root=self.args.data_dir, split='val', transform=self.tr_test)
        np.random.seed(10)

        train_loader = DataLoader(
            trainset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=True,
            **kwargs
        )

        np.random.seed(50)
        val_loader = DataLoader(
            trainset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=True,
            **kwargs
        )

        test_loader = DataLoader(
            testset,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            **kwargs,
        )

        print(
            f"Traing loader: {len(train_loader.dataset)} images, Test loader: {len(test_loader.dataset)} images"
        )
        return train_loader, val_loader, test_loader


class IMAGENET(Dataset):
    def __init__(self, root, num_classes=1000, split='train', transform=None) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        with h5py.File(self.root) as file:
            self.indexes = np.where(np.array(file[self.split]['label']) < num_classes)[0]

    def __len__(self):
        return self.indexes.size

    def __getitem__(self, index):
        with h5py.File(self.root) as file:
            img = torch.Tensor(file[self.split]['data'][self.indexes[index]]).to(torch.uint8)
            label = int(file[self.split]['label'][self.indexes[index]])
        if self.transform != None:
            img = self.transform(img)
        return img, label