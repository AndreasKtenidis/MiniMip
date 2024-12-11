import torch
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader


def load_datasets(partition_id, num_partitions: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=32)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, valloader, testloader

# trainloader0, valloader0, _ = load_datasets(0, 2)
# trainloader1, valloader1, _ = load_datasets(1, 2)
#
# from typing import Dict, List, Optional, Tuple
# def getMinMax(valloader) -> None:
#     a = [1,2,3]
#     _min = min([x for x in [torch.min(batch["label"]) for batch in valloader]])
#     _max = max([x for x in [torch.max(batch["label"]) for batch in valloader]])
#     # f=[(batch["label"]) for batch in valloader]
#     # for batch in trainloader:
#     #     labels = batch["label"]
#     return _min,_max
#
# _min,_max = getMinMax(valloader0)
# print(_min)
# print(_max)


