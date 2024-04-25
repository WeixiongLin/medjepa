# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = []
        # raise RuntimeError(len(cfgs_mask), cfgs_mask)
        for m in cfgs_mask:
            mask_generator = _MaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                spatial_pred_mask_scale=m.get('spatial_scale'),
                temporal_pred_mask_scale=m.get('temporal_scale'),
                aspect_ratio=m.get('aspect_ratio'),
                npred=m.get('num_blocks'),
                max_context_frames_ratio=m.get('max_temporal_keep', 1.0),
                max_keep=m.get('max_keep', None),
            )
            self.mask_generators.append(mask_generator)

    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):

        batch_size = len(batch)
        #x1= batch[0]
        ## x1, x2, x3, x4: all tuple; batch is a list
        ## tuple of 3 eles
        ## x11, x12, x13 = x1  # x11, x13 is list len=1, x12 is int
        ## x11[0] = tensor(3, 16, 224, 224)
        ## x13[0] = tensor(16)
        ## raise RuntimeError(x11[0].shape, x12, x13[0].shape)

        vision_xs = [item[0][0] for item in batch]
        vision_xs = torch.nn.utils.rnn.pad_sequence(
            vision_xs, batch_first=True, padding_value=0
        )
        shapes = [x.shape for x in vision_xs]
        # raise RuntimeError(shapes)
        for i in range(len(vision_xs)):
            batch[i][0][0] = vision_xs[i]

        """
        x1, x2, x3, x4 = batch
        x11, x12, x13 = x1
        x21, x22, x23 = x2
        x31, x32, x33 = x3
        x41, x42, x43 = x4
        """
        # raise RuntimeError(x12, x22, x32, x42)
        # raise RuntimeError(x13.shape, x23.shape, x33.shape, x43.shape)  # 64, 64, 64, 36
        # raise RuntimeError(x11[0].shape, x21[0].shape, x31[0].shape, x41[0].shape)
        # raise RuntimeError( len(x11), x11[0].shape, len(x21), x21[0].shape, len(x31), x31[0].shape, len(x41), x41[0].shape )

        # batch = [(x11), (x21), (x31), (x41)]
        # batch = [(x12), (x22), (x32), (x42)]
        # batch = [(x13), (x23), (x33), (x43)]
        collated_batch = torch.utils.data.default_collate(batch)
        ## collated_batch[0][0] = tensor(4, 3, 16, 224, 224)

        collated_masks_pred, collated_masks_enc = [], []

        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc, masks_pred = mask_generator(batch_size)
            # masks_enc = shape(B, 472)
            # masks_pred = shape(B, 808)
            # raise RuntimeError(masks_enc.shape, masks_pred.shape)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)
        # end for

        return collated_batch, collated_masks_enc, collated_masks_pred


class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))  # maximum number of time-steps (frames) spanned by context mask
        self.max_keep = max_keep  # maximum number of patches to keep in context
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self,
        generator,
        temporal_scale,
        spatial_scale,
        aspect_ratio_scale
    ):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        """
        mask = torch.ones(duration, height, width)
        where there's a zero block of b_size
        """
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start:start+t, top:top+h, left:left+w] = 0

        # Context mask will only span the first X frames
        # (X=self.max_context_frames)
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration:, :, :] = 0

        # --
        return mask

    def __call__(self, batch_size):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        ## raise RuntimeError(self.temporal_pred_mask_scale, self.spatial_pred_mask_scale, self.aspect_ratio)
        ## [1.0, 1.0], [0.15, 0.15], [0.75, 1.5]

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        for _ in range(batch_size):

            empty_context = True
            while empty_context:

                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    # sam_mask = self._sample_block_mask(p_size)
                    # raise RuntimeError(p_size, self.duration, self.width, self.height, sam_mask.shape)
                    mask_e *= self._sample_block_mask(p_size)
                # end for
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)
                # end if
            # end while
        # end for

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        # collated_masks_enc = [tensor(472)*B]
        # collated_masks_pred = [tensor(808)*B]
        # raise RuntimeError( collated_masks_enc[1].shape, collated_masks_pred[1].shape )
        return collated_masks_enc, collated_masks_pred

