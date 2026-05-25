import typing
from typing import Optional, Tuple

from omegaconf import DictConfig
from torch import Generator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .punet import get_dataset


def save_iter(dataloader: DataLoader, sampler: Optional[DistributedSampler] = None) -> typing.Iterator:
    """Return a save iterator over the loader, which supports multi-gpu training using a distributed sampler.

    Args:
        dataloader (DataLoader): DataLoader object.
        sampler (Optional[DistributedSampler]): DistributedSampler object.

    Returns:
        typing.Iterator: Iterator object containing data.
    """
    iterator = iter(dataloader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            if sampler is not None:
                sampler.set_epoch(sampler.epoch + 1)
            yield next(iterator)


def get_dataloader(
    opt: DictConfig, sampling: bool = False
) -> Tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]:
    """
    Return the training and testing dataloaders for the given configuration.

    Args:
        opt (DictConfig): Configuration dictionary.
        sampling (bool): Whether to use sampling.

    Returns:
        Tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]: Training and testing dataloaders.
    """
    test_dataset = None
    collate_fn = None

    if opt.data.dataset == "PUNet":
        train_dataset = get_dataset(
            dataset_root=opt.data.data_dir,
            split="train",
        )
        test_dataset = get_dataset(
            dataset_root=opt.data.data_dir,
            split="test",
        )
    else:
        raise NotImplementedError(f"Dataset {opt.data.dataset} not implemented!")

    # setup the samplers
    if opt.distribution_type == "multi":
        train_sampler = (
            DistributedSampler(train_dataset, num_replicas=opt.global_size, rank=opt.local_rank)
            if train_dataset is not None
            else None
        )
        test_sampler = (
            DistributedSampler(test_dataset, num_replicas=opt.global_size, rank=opt.local_rank)
            if test_dataset is not None
            else None
        )
    else:
        train_sampler = None
        test_sampler = None

    # setup the dataloaders
    train_dataloader = (
        DataLoader(
            train_dataset,
            batch_size=opt.training.bs if not sampling else opt.sampling.bs,
            sampler=train_sampler,
            shuffle=train_sampler is None,
            num_workers=int(opt.data.workers),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        if train_dataset is not None
        else None
    )

    test_dataloader = (
        DataLoader(
            test_dataset,
            batch_size=opt.training.bs if not sampling else opt.sampling.bs,
            sampler=test_sampler,
            shuffle=False,
            num_workers=int(opt.data.workers),
            pin_memory=True,
            drop_last=False,
            generator=Generator().manual_seed(opt.training.seed),
            collate_fn=collate_fn,
        )
        if test_dataset is not None
        else None
    )

    return train_dataloader, test_dataloader, train_sampler, test_sampler
