# TAKEN FROM https://github.com/lufficc/SSD
import logging
import os
import torch
import pathlib
import sys

from torch.hub import download_url_to_file
from torch.hub import urlparse
from torch.hub import HASH_REGEX

class CheckPointer:
    _last_checkpoint_name = 'last_checkpoint.txt'

    def __init__(self,
                 model,
                 optimizer=None,
                 save_dir="",
                 save_to_disk=None,
                 logger=None,
                 scheduler=None,):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data['model'] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found.")
            return {}

        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        model = self.model

        model.load_state_dict(checkpoint.pop("model"))
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        return os.path.exists(save_file)

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, self._last_checkpoint_name)
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        return torch.load(f, map_location=torch.device("cpu"))



# very similar to https://github.com/pytorch/pytorch/blob/master/torch/utils/model_zoo.py
# but with a few improvements and modifications
def cache_url(url, model_dir=None, progress=True):
    r"""Loads the Torch serialized object at the given URL.
    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.
    The default value of `model_dir` is ``$TORCH_HOME/models`` where
    ``$TORCH_HOME`` defaults to ``~/.torch``. The default directory can be
    overridden with the ``$TORCH_MODEL_ZOO`` environment variable.
    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        progress (bool, optional): whether or not to display a progress bar to stderr
    Example:
        >>> cached_file = maskrcnn_benchmark.utils.model_zoo.cache_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        model_dir = os.getenv("TORCH_MODEL_ZOO", pathlib.Path(torch_home, "models"))
    model_dir.mkdir(exist_ok=True, parents=True)
    parts = urlparse(url)
    filename = pathlib.Path(parts.path).name
    if filename == "model_final.pkl":
        # workaround as pre-trained Caffe2 models from Detectron have all the same filename
        # so make the full path the filename by replacing / with _
        filename = parts.path.replace("/", "_")
    cached_file = pathlib.Path(model_dir, filename)
    if cached_file.is_file():
        return cached_file
    sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
    hash_prefix = HASH_REGEX.search(filename)
    if hash_prefix is not None:
        hash_prefix = hash_prefix.group(1)
        # workaround: Caffe2 models don't have a hash, but follow the R-50 convention,
        # which matches the hash PyTorch uses. So we skip the hash matching
        # if the hash_prefix is less than 6 characters
        if len(hash_prefix) < 6:
            hash_prefix = None
    download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file


def load_state_dict_from_url(url, map_location='cpu'):
    cached_file = cache_url(url)
    return torch.load(cached_file, map_location=map_location)