# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torchvision.transforms as transforms

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.image3d.transforms as image3d_transforms
from src.datasets.utils.video.randerase import RandomErasing


def make_transforms(
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3/4, 4/3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    target_depth=64,
    crop_size=224,
    normalize=((0.485, 0.456, 0.406),
               (0.229, 0.224, 0.225))
):

    _frames_augmentation = VideoTransform(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        target_depth=target_depth,
        crop_size=crop_size,
        normalize=normalize,
    )
    return _frames_augmentation


class VideoTransform(object):

    def __init__(
        self,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3/4, 4/3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        target_depth=64,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
    ):

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.target_depth = target_depth
        self.crop_size = crop_size
        self.mean = torch.tensor(normalize[0], dtype=torch.float32)
        self.std = torch.tensor(normalize[1], dtype=torch.float32)
        if not self.auto_augment:
            # Without auto-augment, PIL and tensor conversions simply scale uint8 space by 255.
            self.mean *= 255.
            self.std *= 255.

        self.autoaug_transform = image3d_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

        self.spatial_transform = image3d_transforms.random_resized_crop_with_shift \
            if motion_shift else image3d_transforms.random_resized_crop

        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode='pixel',
            max_count=1,
            num_splits=1,
            device='cpu',
        )

    def __call__(self, buffer):

        if self.auto_augment:
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
            buffer = self.autoaug_transform(buffer)
            buffer = [transforms.ToTensor()(img) for img in buffer]
            buffer = torch.stack(buffer)  # T C H W
            buffer = buffer.permute(0, 2, 3, 1)  # T H W C
        else:
            buffer = torch.tensor(buffer, dtype=torch.float32)

        # buffer = shape[2, 3, 512, 512, 64]  X C H W T
        # buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W
        buffer = buffer.permute(0, 1, 4, 2, 3)  # X C H W T-> X C T H W

        # raise RuntimeError(self.spatial_transform)
        buffer = self.spatial_transform(
            images=buffer,
            target_depth=self.target_depth,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )
        if buffer.shape != torch.Size([3, 2, 64, 64, 64]):
            raise RuntimeError( buffer.shape )
        if self.random_horizontal_flip:
            buffer, _ = image3d_transforms.horizontal_flip(0.5, buffer)

        # mean: [123.6750, 116.2800, 103.5300]
        # std: [58.3950, 57.1200, 57.3750]
        # reprob: 0
        """
        buffer = _tensor_normalize_inplace(buffer, self.mean, self.std)
        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)
        """
        # buffer=[2, 3, 64, 224, 224]
        buffer = buffer.permute(1, 0, 2, 3, 4)
        # buffer=[3, 2, 64, 224, 224]
        # raise RuntimeError(buffer.shape)
        return buffer


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _tensor_normalize_inplace(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize (with dimensions C, T, H, W).
        mean (tensor): mean value to subtract (in 0 to 255 floats).
        std (tensor): std to divide (in 0 to 255 floats).
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()

    # tensor.shape = [2, 3, 64, 224, 224]
    X, C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)  # Make C the last dimension
    tensor.sub_(mean).div_(std)
    tensor = tensor.permute(1, 0).view(C, T, H, W)  # Put C back in front
    return tensor
