import torchvision
import torch
import os

cifar10_standard_transforms = [
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    ),
]
cifar10_augmentations = [
    torchvision.transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomApply(
        [torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        p=0.8,
    ),
    torchvision.transforms.RandomGrayscale(p=0.2),
    *cifar10_standard_transforms,
]


def get_cifar10_dataset(train: bool = True, data_path: str = "data/raw"):
    ds = torchvision.datasets.CIFAR10(data_path, train=train, download=True)
    return ds


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transforms: torchvision.transforms.transforms.Compose,
    ):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        augmented_image = self.transforms(image)
        return augmented_image, label


class SimSiamDataset(AugmentedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        transforms: torchvision.transforms.transforms.Compose,
    ):
        super().__init__(dataset, transforms)
        self.standard_transforms = torchvision.transforms.Compose(
            cifar10_standard_transforms
        )

    def __getitem__(self, i):
        image, label = self.dataset[i]
        aug1 = self.transforms(image)
        aug2 = self.transforms(image)
        image = self.standard_transforms(image)
        return aug1, aug2, image, label


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k, x]


train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomApply(
            [
                torchvision.transforms.ColorJitter(
                    0.4, 0.4, 0.4, 0.1
                )  # not strengthened
            ],
            p=0.8,
        ),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)
base_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
