import logging
import pathlib
import torch
import numpy as np

from codeslam.config.default import cfg
from codeslam.models import make_model
from codeslam.trainer.trainer import start_train
from codeslam.utils.checkpointer import CheckPointer
from codeslam.utils.parser import get_parser
from codeslam.utils.logger import setup_logger
from codeslam.utils import torch_utils
from codeslam.data.build import make_data_loader
from codeslam.trainer.optimizer import make_optimizer
from codeslam.trainer.scheduler import LinearMultiStepWarmUp

np.random.seed(0)
torch.manual_seed(0)

# Allow torch/cudnn to optimize/analyze the input/output shape of convolutions
# To optimize forward/backward pass.
# This will increase model throughput for fixed input shape to the network
torch.backends.cudnn.benchmark = True

# Cudnn is not deterministic by default. Set this to True if you want
# to be sure to reproduce your results
torch.backends.cudnn.deterministic = True

# For debugging
torch.autograd.set_detect_anomaly(True)

def train(cfg):
    logger = logging.getLogger('CodeSLAM.trainer')

    # Load model
    model = make_model(cfg)
    model = torch_utils.to_cuda(model)
    
    # Load data
    data_loader = make_data_loader(cfg, is_train=True, shuffle=True, max_iter=cfg.TRAINER.MAX_ITER)
    
    # Optimizer
    optimizer = make_optimizer(cfg, model)

    # Learning rate scheduler
    scheduler = LinearMultiStepWarmUp(cfg, optimizer) if cfg.TRAINER.OPTIMIZER.GAMMA > 0 else None

    # Checkpointer for saving model during and after training
    checkpointer = CheckPointer(
        model, optimizer, save_dir=cfg.OUTPUT_DIR_MODEL, save_to_disk=True, logger=logger,
    )

    arguments = {"iteration": 0}
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)
    
    # Begin training
    model = start_train(
        cfg, model, data_loader, optimizer,
        checkpointer, arguments, scheduler
    )

    return model
    

def main():
    # Parse config file
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Make directory for outputs (trained models, etc.)
    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir_model = pathlib.Path(cfg.OUTPUT_DIR_MODEL)
    output_dir_model.mkdir(exist_ok=True, parents=True)

    # Setup logger for displaying information during training
    logger = setup_logger("CodeSLAM", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    
    # Train
    model = train(cfg)


if __name__ == '__main__':
    main()

    

    