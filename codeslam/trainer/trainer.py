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
from codeslam.trainer import inference


def start_train(cfg, model, data_loader, optimizer, checkpointer, arguments, scheduler=None):
    logger = logging.getLogger('CodeSLAM.trainer')
    logger.info("Start training ...")
    meters = MetricLogger()

    # Tensorboard
    summary_writer = torch.utils.tensorboard.SummaryWriter(
        log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))

    batch_start = time.time()

    global_step = arguments["global_step"]
    start_iter = arguments["iteration"]
    start_epoch = arguments["epoch"]

    # KL annealing and weighting
    milestone = cfg.TRAINER.KL_MILESTONE

    max_iter = cfg.TRAINER.MAX_ITER
    epochs = cfg.TRAINER.EPOCHS

    if epochs > 1:
        max_iter = len(data_loader)*epochs
        start_iter = 0
        milestone = len(data_loader)*cfg.TRAINER.KL_MILESTONE
    
    try:
        for epoch in range(start_epoch, epochs):
            epoch = epoch + 1
            for iteration, (images, targets) in enumerate(data_loader, start_iter):
                iteration = iteration + 1
                global_step = global_step + 1
                arguments["iteration"] = iteration
                arguments["global_step"] = global_step

                # Load data to GPU if available
                images = torch_utils.to_cuda(images)
                targets = torch_utils.to_cuda(targets)

                # Forward
                loss_dict = model(images, targets)

                # Weighting
                weights = dict(kl_div=weighting.beta(global_step, milestone))
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
                meters.update(time=batch_time)
                if cfg.TRAINER.LOG_STEP > 0 and global_step % cfg.TRAINER.LOG_STEP == 0:
                    # Logging to terminal and file
                    meters.update(
                        total_loss = loss,
                        kl_div = unweighted_loss_dict["kl_div"],
                        beta = weighting.beta(global_step, milestone)
                    )
                    meters.update(time=batch_time)
                    eta_seconds = meters.time.global_avg * (max_iter - global_step)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    to_log = [
                        f'epoch: {epoch}/{epochs}',
                        f'iter: {global_step:06d}/{max_iter}',
                        f"lr: {optimizer.param_groups[0]['lr']:.5f}",
                        f'{meters}',
                        f"eta: {eta_string}",
                    ]
                    if torch.cuda.is_available():
                        mem = round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                        to_log.append(f'mem: {mem}M')
                    logger.info(meters.delimiter.join(to_log))

                    # Tensorboard
                    summary_writer.add_scalar('losses/total_loss', loss, global_step=global_step)
                    # summary_writer.add_scalar('losses/unweighted_reconstruction_loss', model.unweighted_reconstruction_loss, global_step=global_step)
                    for loss_name, loss_item in unweighted_loss_dict.items():
                        summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                    summary_writer.add_scalar('parameters/lr', optimizer.param_groups[0]['lr'], global_step=global_step)
                    summary_writer.add_scalar('parameters/beta', weighting.beta(global_step, milestone), global_step=global_step)
                    summary_writer.add_scalar('uncertainty/mean', torch.mean(model.b), global_step=global_step)
                    summary_writer.add_scalar('uncertainty/median', torch.median(model.b), global_step=global_step)
                
                # Save model
                if cfg.TRAINER.SAVE_STEP > 0 and global_step % cfg.TRAINER.SAVE_STEP == 0:
                    checkpointer.save("model_{:06d}".format(global_step), **arguments)

                # Validation
                if cfg.TRAINER.EVAL_STEP > 0 and global_step % cfg.TRAINER.EVAL_STEP == 0:
                    model.eval()
                    eval_results = inference.do_evaluation(cfg, model, global_step=global_step)
                    model.train()

                    for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                        for key in eval_result:
                            summary_writer.add_scalar('metrics/{}'.format(key), eval_result[key], global_step=global_step)
                
            arguments["epoch"] = epoch

    except KeyboardInterrupt:
        print(f'\n\nCtrl-c, saving current model ...')
        checkpointer.save("model_{:06d}".format(global_step), **arguments)
        sys.exit()
        
    checkpointer.save("model_final", **arguments)   
    return model

