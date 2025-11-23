import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms as T
import multiprocessing
from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = 1
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.
    """

    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    data_transforms = {
        "train": T.Compose(
            [
                T.Resize(256),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.RandomRotation(30),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                T.RandomApply([T.GaussianBlur(kernel_size=23)], p=0.2),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        ),
        "valid": T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        ),
    }

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms["train"],
    )

    valid_data = datasets.ImageFolder(
        base_path / "valid",
        transform=data_transforms["valid"],
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    # create the test data loader
    test_data = datasets.ImageFolder(
        base_path / "test",
        transform=data_transforms["test"],
    )

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data
    """

    # obtain one batch of training images
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = T.Compose(
        [
            T.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            T.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # Get class names from the train data loader
    class_names = data_loaders["train"].dataset.classes

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()])
