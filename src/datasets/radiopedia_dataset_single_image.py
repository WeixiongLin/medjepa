# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings

from logging import getLogger

import numpy as np
import pandas as pd
import json
import random


from decord import VideoReader, cpu

import torch

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_radiopediadataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
):
    dataset = RadiopediaDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform)

    logger.info('VideoDataset dataset created')
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('VideoDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler


def stack_images(images):
    target_H = 512
    target_W = 512
    target_D = 4
   
    if len(images) == 0:
        return torch.zeros((1,3,target_H,target_W,target_D))
    MAX_D = 4
    D_list = list(range(4,65,4))
    
    for ii in images:
        try:
            D = ii.shape[3]
            if D > MAX_D:
                MAX_D = D
        except:
            continue
    for temp_D in D_list:
        if abs(temp_D - MAX_D)< abs(target_D - MAX_D):
            target_D = temp_D
    
    stack_images = []
    for s in images:
        if len(s.shape) == 3:
        #print(s.shape)
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0).unsqueeze(-1), size = (target_H,target_W,target_D)))
        else:
            stack_images.append(torch.nn.functional.interpolate(s.unsqueeze(0), size = (target_H,target_W,target_D)))
    images = torch.cat(stack_images, dim=0)
    return images


class RadiopediaSingleImageDataset(torch.utils.data.Dataset):
    """ Video classification dataset. """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths and labels
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:

            if data_path[-4:] == '.csv':
                data = pd.read_csv(data_path, header=None, delimiter=" ")
                samples += list(data.values[:, 0])
                labels += list(data.values[:, 1])
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

            elif data_path[-4:] == '.npy':
                data = np.load(data_path, allow_pickle=True)
                data = list(map(lambda x: repr(x)[1:-1], data))
                samples += data
                labels += [0] * len(data)
                num_samples = len(data)
                self.num_samples_per_dataset.append(len(data))

            elif data_path[-5:] == '.json':
                with open(data_path, 'r') as f:
                    data = json.load(f)

                # data_index = data[0]
                # img_path = data_index['npy_path']
                # finding = data_index['finding']
                # impression = data_index['impression']
                # caption = data_index['image_caption']
                # modality = data_index['image_modality'][0]

                samples = [datum['npy_path'] for datum in data]  # str
                labels = [0 for _ in range(len(samples))]  # int, Optional
                # raise RuntimeError( samples[0], labels[0], )
                self.num_samples_per_dataset.append(len(data))

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample = self.samples[index]

        # Keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            # sample = str: /path/to/sample_0004.mp4
            # buffer, clip_indices = self.loadvideo_decord(sample)
            buffer, clip_indices = self.load_npy(sample)
            raise RuntimeError(len(buffer))

            # buffer: List len=16, clip_indices: List len=1
            # buffer[0] = shape(320, 426, 3), clip_indices = shape(16)
            # raise RuntimeError(len(buffer), buffer[0].shape, len(clip_indices), clip_indices[0].shape)
            loaded_video = len(buffer) > 0
            if not loaded_video:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)

        # buffer = [2, 3, 512, 512, 64]
        buffer = split_into_clips(buffer)
        # buffer: List of 1 element, whose shape [2, 3, 512, 512, 64]
        # raise RuntimeError(len(buffer), buffer[0].shape)

        # raise RuntimeError(self.transform)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]
            # buffer: List len=1, buffer[0]=[2, 3, 64, 224, 224]
            # raise RuntimeError(len(buffer), buffer[0].shape)

        # buffer: List len=1, label: int, clip_indices: List len=1
        # raise RuntimeError(len(buffer), label, len(clip_indices))
        # buffer[0] = shape([3, 16, 224, 224])
        # label =  0
        # clip_indices[0] = shape(16)
        return buffer, label, clip_indices

    def load_npy(self, sample):
        buffer = np.load(sample)
        # (3, 3, 512, 512, 66)
        # raise RuntimeError(buffer.shape)

        images = buffer
        ref_image = []
        for index, image in enumerate(images):
            # raise RuntimeError(image.shape)
            # image = shape((3, 512, 512, 66)
            image = (image-image.min())/(image.max()-image.min())
            image = torch.from_numpy(image).float()
            ref_image.append(image)
        # end for

        if len(ref_image) > 2:
            ref_image = random.sample(ref_image, 2)

        vision_x = stack_images(ref_image)  # [2, 3, 512, 512, 64]
        vision_x = vision_x.permute(1, 0, 2, 3, 4)  # [3, 2, 512, 512, 64]
        # end if
        # raise RuntimeError(vision_x.shape)

        # depth = vision_x.shape[-1]
        clip_indices = torch.zeros(64)
        return vision_x, clip_indices

    def loadvideo_decord(self, sample):
        """ Load video content using Decord """

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f'video path not found {fname=}')
            return [], None

        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short {fname=}')
            return [], None
        if _fsize > self.filter_long_videos:
            warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None

        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
        except Exception:
            return [], None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            try:
                fps = vr.get_avg_fps()
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                warnings.warn(e)
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f'skipping video of length {len(vr)}')
            return [], None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                    indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                    indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)
