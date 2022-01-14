from torch.utils.data import ConcatDataset

from codeslam.config.path_catlog import DatasetCatalog
from .scenenet import SceneNetDataset

_DATASETS = {
    'SceneNetRGBD': SceneNetDataset
}


def build_dataset(base_path: str, dataset_list, transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(base_path, dataset_name)
        args = data['args']
        factory = _DATASETS[data['factory']]
        args['transform'] = transform
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
    return [dataset]
