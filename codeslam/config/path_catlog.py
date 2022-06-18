import os

class DatasetCatalog:
    DATA_DIR = ''
    DATASETS = {
        'SceneNetRGBD_train_0': {
            'data_dir': "SceneNetRGBD/train_0/train/",
            'split': "train"
        },
        'SceneNetRGBD_train_16': {
            'data_dir': "SceneNetRGBD/train_16/train/",
            'split': "val"
        }
    }

    @staticmethod
    def get(base_path, name):
        assert name in DatasetCatalog.DATASETS,\
            f"Did not find dataset: {name} in dataset catalog. {DatasetCatalog.DATASETS.keys()}"
        root = os.path.join(base_path, DatasetCatalog.DATA_DIR)
        attrs = DatasetCatalog.DATASETS[name]
        data_dir = os.path.join(root, DatasetCatalog.DATA_DIR, attrs["data_dir"])
        if "SceneNetRGBD" in name:
            args = dict(data_dir=data_dir, split=attrs["split"])
            return dict(factory="SceneNetRGBD", args=args)
        raise RuntimeError("Dataset not available: {}".format(name))
