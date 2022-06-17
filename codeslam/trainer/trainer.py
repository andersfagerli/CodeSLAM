import os
import sys
import logging
import time
import datetime
import torch
import torch.utils.tensorboard

from codeslam.utils.metriclogger import MetricLogger
from codeslam.utils import torch_utils
from codeslam.trainer import weighting


def start_train(cfg, model, data_loader, optimizer, checkpointer, arguments, scheduler=None):
    logger = logging.getLogger('CodeSLAM.trainer')
    logger.info("Start training ...")
    meters = MetricLogger()

    # Tensorboard
    summary_writer = torch.utils.tensorboard.SummaryWriter(
        log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))

    batch_start = time.time()
    max_iter = cfg.TRAINER.MAX_ITER
    start_iter = arguments["iteration"]

    # KL annealing and weighting
    milestone = cfg.TRAINER.KL_MILESTONE
    
    try:
        for iteration, (images, targets) in enumerate(data_loader, start_iter):
            iteration = iteration + 1
            arguments["iteration"] = iteration

            # Load data to GPU if available
            images = torch_utils.to_cuda(images)
            targets = torch_utils.to_cuda(targets)

            # Forward
            loss_dict = model(images, targets)

            # Weighting
            weights = dict(kl_div=weighting.beta(iteration, milestone))
            loss_dict, unweighted_loss_dict = weighting.weigh(loss_dict, weights)

            # Total loss
            loss = sum(loss for loss in loss_dict.values())

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Calculate batch time for logging
            batch_time = time.time() - batch_start
            batch_start = time.time()
            
            # Logging
            if cfg.TRAINER.LOG_STEP > 0 and iteration % cfg.TRAINER.LOG_STEP == 0:
                # Logging to terminal and file
                meters.update(
                    total_loss = loss,
                    recon_loss = model.unweighted_reconstructioin_loss,
                    kl_div = unweighted_loss_dict["kl_div"],
                    beta = weighting.beta(iteration, milestone)
                )
                meters.update(time=batch_time)
                eta_seconds = meters.time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                to_log = [
                    f'iter: {iteration:06d}/{max_iter}',
                    f"lr: {optimizer.param_groups[0]['lr']:.5f}",
                    f'{meters}',
                    f"eta: {eta_string}",
                ]
                if torch.cuda.is_available():
                    mem = round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                    to_log.append(f'mem: {mem}M')
                logger.info(meters.delimiter.join(to_log))

                # Tensorboard
                summary_writer.add_scalar('losses/total_loss', loss, global_step=iteration)
                summary_writer.add_scalar('losses/unweighted_reconstruction_loss', model.unweighted_reconstruction_loss, global_step=iteration)
                for loss_name, loss_item in unweighted_loss_dict.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=iteration)
                summary_writer.add_scalar('parameters/lr', optimizer.param_groups[0]['lr'], global_step=iteration)
                summary_writer.add_scalar('parameters/beta', weighting.beta(iteration, milestone), global_step=iteration)
                summary_writer.add_scalar('uncertainty/mean', torch.mean(model.b), global_step=iteration)
                summary_writer.add_scalar('uncertainty/median', torch.median(model.b), global_step=iteration)
            
            # Save model
            if cfg.TRAINER.SAVE_STEP > 0 and iteration % cfg.TRAINER.SAVE_STEP == 0:
                checkpointer.save("model_{:06d}".format(iteration), **arguments)

    except KeyboardInterrupt:
        print(f'\n\nCtrl-c, saving current model ...')
        checkpointer.save("model_{:06d}".format(iteration), **arguments)
        sys.exit()
        
    checkpointer.save("model_final", **arguments)   
    return model

