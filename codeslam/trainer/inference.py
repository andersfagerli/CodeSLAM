import os
import pathlib
import logging
import torch
import numpy as np

from tqdm import tqdm
from datetime import datetime

from codeslam.utils import torch_utils
from codeslam.data.build import make_data_loader
from codeslam.trainer import metrics


@torch.no_grad()
def do_evaluation(cfg, model, **kwargs):
    data_loaders_val = make_data_loader(cfg, is_train=False)
    eval_results = []
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = pathlib.Path(cfg.OUTPUT_DIR, "inference", dataset_name)
        output_folder.mkdir(exist_ok=True, parents=True)

        eval_result = inference(cfg, model, data_loader, dataset_name)
        eval_results.append(eval_result)

        log(eval_result, output_folder, **kwargs)

    return eval_results


def inference(cfg, model, data_loader, dataset_name):
    dataset = data_loader.dataset

    logger = logging.getLogger("CodeSLAM.inference")
    logger.info(
        "Evaluating {} dataset({} images):".format(dataset_name, len(dataset)))
    
    eval_results = compute_on_dataset(cfg, model, data_loader)
    
    return eval_results


def compute_on_dataset(cfg, model, data_loader):
    N = 1000 # len(dataloader) to evaluate over entire dataloader
    metric = np.empty((N, 7))    # (batches x metrics)
    
    for i, batch in enumerate(tqdm(data_loader)):
        if i >= N:
            break

        images, targets = batch
        images = torch_utils.to_cuda(images)

        with torch.no_grad():
            result = model(images)

        depths = torch_utils.to_numpy(result["depth"])
        targets = torch_utils.to_numpy(targets)

        metric[i,:] = evaluate(depths, targets)
    
    metric = np.mean(metric, axis=0)
    return {
        "rmse":     metric[0],
        "logrmse":  metric[1],
        "ard":      metric[2],
        "srd":      metric[3],
        "a1":       metric[4],
        "a2":       metric[5],
        "a3":       metric[6]
    }


def evaluate(predictions, ground_truth):
    rmse = metrics.RMSE(predictions, ground_truth)
    logrmse = metrics.logRMSE(predictions, ground_truth)
    ard = metrics.ARD(predictions, ground_truth)
    srd = metrics.SRD(predictions, ground_truth)
    a1 = metrics.accuracy(predictions, ground_truth, threshold=1.25**1)
    a2 = metrics.accuracy(predictions, ground_truth, threshold=1.25**2)
    a3 = metrics.accuracy(predictions, ground_truth, threshold=1.25**3)

    return np.array([rmse, logrmse, ard, srd, a1, a2, a3])
    
    
def log(eval_result, output_folder, global_step=None):
    logger = logging.getLogger("CodeSLAM.inference")
    result_str = "rmse: {:.4f} logrmse: {:.4f} ard: {:.4f} srd: {:.4f} a1: {:.4f} a2: {:.4f} a3: {:.4f}\n"\
                .format(
                    eval_result["rmse"], eval_result["logrmse"], eval_result["ard"], eval_result["srd"],
                    eval_result["a1"], eval_result["a2"], eval_result["a3"]
                )
    logger.info(result_str)

    if global_step is not None:
        result_path = os.path.join(output_folder, 'result_{:07d}.txt'.format(global_step))
    else:
        result_path = os.path.join(output_folder, 'result_{}.txt'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    with open(result_path, "w") as f:
        f.write(result_str)