#'') Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed

import random
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str, help='name of config file to load', default='configs.yaml')
parser.add_argument('--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine'
)
parser.add_argument('--name', type=str)
parser.add_argument('--resume', type=str, default=None, help='path to latest checkpoint (default: none)')

parser.add_argument('--use_wandb', action="store_true")
parser.add_argument('--wandb_project_name', type=str, default='medjepa')
parser.add_argument('--wandb_notes', type=str, default='')


def process_main(rank, world_size, args):
    fname = args.fname
    devices = args.devices

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    from src.utils.logging import get_logger
    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')

    # Log config
    if rank == 0:
        # 1. local log
        pprint.PrettyPrinter(indent=4).pprint(params)
        dump = os.path.join(params['logging']['folder'], 'params-pretrain.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

        # 2. wandb log
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project_name,
                name=args.name,
                id=args.name,
                notes=args.wandb_notes,
                tags=[],
                resume='auto' if args.resume == "latest" else None,
                config=vars(args),
                mode='offline',
            )
        # end if

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(port=37123 + random.randint(0, 101), rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Launch the app with loaded config
    app_main(params['app'], param_args=params, args=args)


if __name__ == '__main__':
    args = parser.parse_args()
    num_gpus = len(args.devices)
    mp.set_start_method('spawn')
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, num_gpus, args)
        ).start()
