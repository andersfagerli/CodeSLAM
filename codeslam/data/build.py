import torch
from torch.utils.data import DataLoader

from codeslam.data import samplers
from codeslam.data.dataset import build_dataset
from codeslam.data.transforms import build_transforms


def make_data_loader(cfg, is_train=True, shuffle=True, start_iter=0):
    train_transform = build_transforms(cfg, is_train=is_train)
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    max_iter = cfg.TRAINER.MAX_ITER
    epochs = cfg.TRAINER.EPOCHS

    datasets = build_dataset(
        cfg.DATASETS.DATASET_DIR,
        dataset_list, transform=train_transform,
        is_train=is_train)

    data_loaders = []

    for dataset in datasets:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg.TRAINER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=is_train)
        if max_iter is not None and epochs == 1:
            batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter, start_iter=start_iter)

        data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY)
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
