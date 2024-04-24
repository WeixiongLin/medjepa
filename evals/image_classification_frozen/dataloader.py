import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


from src.datasets.data_manager import (
    init_data,
)
from app.medjepa.transforms import make_transforms


def make_dataloader(
    dataset_name,
    root_path,
    label_paths,
    image_folder,
    batch_size,
    world_size,
    rank,
    resolution=224,
    training=False,
    subset_file=None,
    use_aws=None,
    cfgs_data=None,
    data_aug=None,
):
    normalization = ((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))

    # raise RuntimeError( data_aug )
    target_depth = cfgs_data.get('target_depth')
    crop_size = cfgs_data.get('crop_size')

    if training:
        logger.info('implementing auto-agument strategy')
        """
        transform = timm_make_transforms(
            input_size=resolution,
            is_training=training,
            auto_augment='original',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=normalization[0],
            std=normalization[1]
        )
        """

        ar_range = data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
        rr_scale = data_aug.get('random_resize_scale', [0.3, 1.0])
        motion_shift = data_aug.get('motion_shift', False)
        reprob = data_aug.get('reprob', 0.)
        use_aa = data_aug.get('auto_augment', False)

        transform = make_transforms(
            random_horizontal_flip=True,
            random_resize_aspect_ratio=ar_range,
            random_resize_scale=rr_scale,
            reprob=reprob,
            auto_augment=use_aa,
            motion_shift=motion_shift,
            target_depth=target_depth,
            crop_size=crop_size
        )

    else:
        ar_range = data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
        rr_scale = data_aug.get('random_resize_scale', [0.3, 1.0])
        motion_shift = data_aug.get('motion_shift', False)
        reprob = data_aug.get('reprob', 0.)
        use_aa = data_aug.get('auto_augment', False)

        transform = make_transforms(
            random_horizontal_flip=True,
            random_resize_aspect_ratio=ar_range,
            random_resize_scale=rr_scale,
            reprob=reprob,
            auto_augment=use_aa,
            motion_shift=motion_shift,
            target_depth=target_depth,
            crop_size=crop_size
        )
    # end if


    data_loader, _ = init_data(
        data=dataset_name,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        label_paths=label_paths,
        # image_folder=image_folder,
        training=training,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file,
        use_aws=use_aws,
    )
    return data_loader

