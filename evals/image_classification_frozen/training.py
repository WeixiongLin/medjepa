import torch
import logging
from .losses import MultiLabelLoss
from .evaluate import calculate_mAP, calculate_multilabel_AUC, calculate_mF1max_MCC

from src.utils.logging import (
    AverageMeter,
    CSVLogger
)
from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule,
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)



def init_opt(
    classifier,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False
):
    param_groups = [
        {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler



def run_one_epoch(
    device,
    training,
    encoder,
    classifier,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
):

    classifier.train(mode=training)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = MultiLabelLoss()

    top1_meter = AverageMeter()
    all_outputs, all_labels = [], []

    for itr, data in enumerate(data_loader):
        if training:
            scheduler.step()
            wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):

            # raise RuntimeError( type(data), type(data[0]), type(data[1]), type(data[2]) )
            # imgs, labels = data[0].to(device), data[1].to(device)
            imgs, labels, _  = data
            imgs = imgs[0].to(device)
            labels = labels.to(device)

            # raise RuntimeError(imgs.shape, labels.shape)
            with torch.no_grad():
                outputs = encoder(imgs)
                if not training:
                    outputs = classifier(outputs)
            if training:
                outputs = classifier(outputs)
        # end with

        loss = criterion(outputs, labels)
        all_outputs.append(outputs.cpu().detach())
        all_labels.append(labels.cpu().detach())
        # raise RuntimeError(loss, outputs.shape, labels.shape)  # [B, NUM_CLASS]
        # top1_acc = 100. * outputs.max(dim=1).indices.eq(labels).sum() / len(imgs)
        # top1_acc = float(AllReduce.apply(top1_acc))
        # top1_meter.update(top1_acc)

        if training:
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        if itr % 20 == 0:
            logger.info('[%5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                        % (itr, top1_meter.avg, loss,
                           torch.cuda.max_memory_allocated() / 1024.**2))
        # end if
    # end for

    raise RuntimeError( type(all_outputs) )
    all_outputs_tensor = torch.cat(all_outputs)
    all_labels_tensor = torch.cat(all_labels)

    # raise RuntimeError(all_outputs[0].shape, all_labels[0].shape, all_outputs_tensor.shape, all_labels_tensor.shape)
    all_outputs_array, all_labels_array = all_outputs_tensor, all_labels_tensor
    mAP = calculate_mAP(all_labels_array, all_outputs_array)
    auc = calculate_multilabel_AUC(all_labels_array, all_outputs_array)
    f1, mcc = calculate_mF1max_MCC(all_labels_array, all_outputs_array)
    # raise RuntimeError(mAP)

    return {'mAP': mAP, 'auc': auc, 'f1': f1, 'mcc': mcc}
