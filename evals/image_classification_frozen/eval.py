# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint

import numpy as np

import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from torch.nn.parallel import DistributedDataParallel

from timm.data import create_transform as timm_make_transforms

from src.models.attentive_pooler import AttentiveClassifier, LinearClassifier
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    AverageMeter,
    CSVLogger
)
from .model import init_video_model, load_checkpoint
from .dataloader import make_dataloader
from .training import run_one_epoch, init_opt


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    args_data_aug = args_eval.get('data_aug')
    use_aws = args_data.get('use_aws', None)
    dataset_name = args_data.get('dataset_name')
    num_classes = args_data.get('num_classes')

    # root_path = args_data.get('root_path', None)
    dataset_paths = args_data.get('datasets', [])
    label_paths = args_data.get('labels', [])
    image_folder = args_data.get('image_folder', None)
    resolution = args_data.get('resolution', 224)

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    batch_size = args_opt.get('batch_size')
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')

    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None)

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, 'image_classification_frozen/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(log_file,
                               ('%d', 'epoch'),
                               ('%.5f', 'loss'),
                               ('%.5f', 'acc'))

    # Initialize model

    # -- pretrained encoder (frozen)
    """
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        frames_per_clip=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)
    """

    use_mask_tokens = True
    cfgs_mask = [{
        'aspect_ratio': [0.75, 1.5], 'num_blocks': 8, 'spatial_scale': [0.15, 0.15], 'temporal_scale': [1.0, 1.0],
        'max_temporal_keep': 1.0, 'max_keep': None}, {'aspect_ratio': [0.75, 1.5], 'num_blocks': 2,
        'spatial_scale': [0.7, 0.7], 'temporal_scale': [1.0, 1.0], 'max_temporal_keep': 1.0, 'max_keep': None
    }]
    zero_init_mask_tokens = True
    num_frames = 16
    pred_depth = 12
    pred_embed_dim = 384

    # """
    encoder, predictor = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=resolution,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
    )
    # """

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # -- init classifier
    encoder_embed_dim = 1024
    encoder_num_heads = 64
    encoder_num_classes = 64

    """
    classifier = AttentiveClassifier(
        embed_dim=encoder_embed_dim,
        num_heads=encoder_num_heads,
        depth=1,
        num_classes=num_classes
    ).to(device)
    """
    classifier = LinearClassifier(
        batch_size = batch_size,
        embed_dim=encoder_embed_dim,
        num_heads=encoder_num_heads,
        depth=1,
        num_classes=num_classes
    ).to(device)

    train_loader = make_dataloader(
        dataset_name=dataset_name,
        # root_path=root_path,
        root_path=dataset_paths,
        label_paths=label_paths,
        resolution=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        use_aws=use_aws,
        cfgs_data=args_data,
        data_aug=args_data_aug,
    )
    val_loader = make_dataloader(
        dataset_name=dataset_name,
        # root_path=root_path,
        root_path=dataset_paths,
        label_paths=label_paths,
        resolution=resolution,
        image_folder=image_folder,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        use_aws=use_aws,
        cfgs_data=args_data,
        data_aug=args_data_aug,
    )
    ipe = len(train_loader)

    # itr = iter(train_loader)
    # data = next(itr)
    # buffer = data[0]
    # raise RuntimeError( type(buffer), len(buffer), type(buffer[0]) )

    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    # -- optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifier=classifier,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16)
    # classifier = DistributedDataParallel(classifier, static_graph=True)

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint:
        classifier, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch):
        save_dict = {
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        train_metric = run_one_epoch(
            device=device,
            training=True,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16)

        val_metric = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifier=classifier,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16)

        logger.info('[%5d] train: %.3f test: %.3f' % (epoch + 1, train_metric['mAP'], val_metric['mAP']))
        if rank == 0:
            csv_logger.log(epoch + 1, train_metric['mAP'], val_metric['mAP'])
        save_checkpoint(epoch + 1)
    # end for

