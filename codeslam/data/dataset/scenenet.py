import glob
import torch
import torch.utils.data
import numpy as np
from PIL import Image

class SceneNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        """Dataset for SceneNet RGB-D. Assumes RGB images as input (transform) and depth as target (target_transform)
        Args:
            data_dir: the root of the specific SceneNet dataset, the directory contains the following sub-directories:
                depth, instance, photo.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.rgb_list = sorted(glob.glob(data_dir+'photo/*.png'))
        self.depth_list = sorted(glob.glob(data_dir+'depth/*.png'))

        assert len(self.rgb_list) == len(self.depth_list),\
            f"Number of elements in folders \"depth\" and \"photo\" do not match"

    def __getitem__(self, index):
        depth_image = self._read_depth_image(index)
        rgb_image = self._read_rgb_image(index)

        if self.transform:
            rgb_image, depth_image = self.transform(rgb_image, depth_image)
        
        return rgb_image, depth_image

    def __len__(self):
        return len(self.rgb_list)

    def _read_depth_image(self, image_id):
        image_file = self.depth_list[image_id]
        image = Image.open(image_file)
        image = np.array(image)
        return image
    
    def _read_rgb_image(self, image_id):
        image_file = self.rgb_list[image_id]
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
    
    def _read_instance_image(self, image_id):
        raise NotImplementedError


if __name__ == "__main__":
    # Example of how to calculate mean and std for image normalization
    dataset = SceneNetDataset(data_dir='datasets/SceneNetRGBD/train_0/train/')
    
    sample_rgb, sample_depth = dataset[0]

    num_pixels_rgb = sample_rgb.shape[0] * sample_rgb.shape[1] * len(dataset)
    num_pixels_depth = sample_depth.shape[0] * sample_depth.shape[1] * len(dataset)

    sum_rgb = np.array([0.0, 0.0, 0.0])
    sum_depth = 0.0
    for i in range(len(dataset)):
        image, depth = dataset[i]

        sum_rgb += image.sum((0,1))
        sum_depth += depth.sum()
     
    rgb_means = sum_rgb / num_pixels_rgb
    depth_mean = sum_depth / num_pixels_depth

    sum_rgb_squared_error = np.array([0.0, 0.0, 0.0])
    sum_depth_squared_error = 0.0
    for i in range(len(dataset)):
        image, depth = dataset[i]

        sum_rgb_squared_error += ((image - rgb_means)**2).sum((0,1))
        sum_depth_squared_error += ((depth - depth_mean)**2).sum()
    
    rgb_stds = np.sqrt(sum_rgb_squared_error / num_pixels_rgb)
    depth_std = np.sqrt(sum_depth_squared_error / num_pixels_depth)

    print(f'RGB means:\t{rgb_means}')
    print(f'RGB stds:\t{rgb_stds}')
    print(f'Depth mean:\t{depth_mean}')
    print(f'Depth std:\t{depth_std}')
    
    
    