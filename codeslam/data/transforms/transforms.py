# from https://github.com/amdegroot/ssd.pytorch
import torch
import cv2
import numpy as np
import torch.functional as F

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
            
        return img, target

### Image and target transforms ###
class Identity(object):
    def __call__(self, image, target):
        return image, target

class ConvertFromInts(object):
    def __call__(self, image, target=None):
        if target is not None:
            target = target.astype(np.float32)
        return image.astype(np.float32), target

class Resize(object):
    def __init__(self, imshape):
        h, w = imshape
        self.shape = (w, h)

    def __call__(self, image, target=None):
        image = cv2.resize(image.astype(np.float32), self.shape)
        if target is not None:
            target = cv2.resize(target.astype(np.float32), self.shape)
        return image, target

class ToTensor(object):
    def __call__(self, cvimage, target=None):
        if len(cvimage.shape) < 3:
            cvimage = np.expand_dims(cvimage, axis=-1)

        if target is None:
            return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), None
        else:
            if len(target.shape) < 3:
                target = np.expand_dims(target, axis=-1)

            return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1),\
                torch.from_numpy(target.astype(np.float32)).permute(2, 0, 1)

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, target):
        # image/target: (HxWxC) or (HxW)
        if np.random.rand(1) < self.p:
            image = image[:, ::-1, :] if len(image.shape) > 2 else image[:, ::-1]
            target = target[:, ::-1, :] if len(target.shape) > 2 else target[:, ::-1]

        return image, target

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, target):
        # image/target: (HxWxC) or (HxW)
        if np.random.rand(1) < self.p:
            image = image[::-1, :, :] if len(image.shape) > 2 else image[::-1, :]
            target = target[::-1, :, :] if len(target.shape) > 2 else target[::-1, :]

        return image, target

### Image transforms ###
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, target=None):
        image = image.astype(np.float32)

        image = (image - self.mean) / self.std
        return image.astype(np.float32), target

class RGBToGrayscale(object):
    def __call__(self, image, target=None):
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    # (HxW)
        image_gray = np.expand_dims(image_gray, axis=-1)        # (HxWxC)
        return image_gray, target

class Scale(object):
    def __call__(self, image, target=None):
        return image * (1.0/255), target

### Target transforms ###
class Proximity(object):
    """
    Transforms depth values to the range [0,1] as described in https://arxiv.org/pdf/1804.00874.pdf
    """
    def __init__(self, mean):
        self.a = mean
    
    def __call__(self, image, target):
        inverse_proximity = (target.astype(np.float32) + self.a) * (1.0/self.a)
        target = 1.0/inverse_proximity
        return image, target

class ProximityToDepth(object):
    """
    Transforms depth values from the proximity range [0,1] back to the original [0, max]
    """
    def __init__(self, mean, max):
        self.a = mean
        self.max = max
    
    def __call__(self, image, target, eps=1e-06):
        target = np.clip(self.a * (1.0-target) / np.clip(target, a_min=eps, a_max=None), a_min=0, a_max=self.max)
        return image, target




