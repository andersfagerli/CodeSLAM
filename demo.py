import os
import glob
import pathlib
import time
import numpy as np
from PIL import Image
import torch

from codeslam.config.default import cfg
from codeslam.models import make_model
from codeslam.utils.checkpointer import CheckPointer
from codeslam.utils.parser import get_parser
from codeslam.utils import torch_utils
from codeslam.data.transforms import build_transforms
from codeslam.data.transforms.transforms import ProximityToDepth, Resize
from utils.plotting import plot_comparison

@torch.no_grad()
def run_demo(cfg):
    model = make_model(cfg)
    model = torch_utils.to_cuda(model)

    ckpt = cfg.PRETRAINED_WEIGHTS if len(cfg.PRETRAINED_WEIGHTS) > 0 else None 
    if ckpt is None:
        raise RuntimeError("Specify file with model weights in config")
    
    demo_dir = cfg.OUTPUT_DIR + '/demo'
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(ckpt, use_latest=ckpt is None)
    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()
    print('Loaded weights from {}'.format(weight_file))

    image_paths_photo = sorted(glob.glob(os.path.join(cfg.DEMO_RGB_PATH, '*.png')))
    image_paths_depth = sorted(glob.glob(os.path.join(cfg.DEMO_DEPTH_PATH, '*.png')))

    transforms = build_transforms(cfg, is_train=False)
    proximity_to_depth_transform = ProximityToDepth(cfg.OUTPUT.PIXEL_MEAN, cfg.OUTPUT.PIXEL_MAX)
    resize = Resize(cfg.INPUT.IMAGE_SIZE)
    
    model.eval()

    inference_time = 0
    comparisons = []
    for i, (image_path_photo, image_path_depth) in enumerate(zip(image_paths_photo, image_paths_depth)):
        photo = np.array(Image.open(image_path_photo).convert("RGB"))
        depth_gt = np.array(Image.open(image_path_depth))

        photo, depth_gt = resize(photo, depth_gt)

        photo_transformed, depth_gt_transformed = transforms(photo, depth_gt)

        photo_transformed = photo_transformed.unsqueeze(0)
        depth_gt_transformed = depth_gt_transformed.unsqueeze(0)
        
        photo_transformed = torch_utils.to_cuda(photo_transformed)
        depth_gt_transformed = torch_utils.to_cuda(depth_gt_transformed)

        start = time.time()

        with torch.no_grad():
            result = model(photo_transformed)
        
        inference_time += time.time() - start

        depth = result["depth"]
        depth = torch_utils.to_numpy(depth)

        b = result["b"]
        b = torch_utils.to_numpy(b)

        _, depth = proximity_to_depth_transform(photo_transformed, depth)

        comparisons.extend([[np.int32(photo), depth_gt, depth.squeeze((0,1)), b.squeeze((0,1))]])

    plot_comparison(comparisons)
        
    print(f'Average inference time: {inference_time/len(comparisons)}')


def main():
    # Parse config file
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = pathlib.Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    run_demo(cfg)


if __name__ == '__main__':
    main()