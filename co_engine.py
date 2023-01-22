# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import torch.nn.functional as F
import utils
import torch.nn as nn

def loss_kd(preds, labels, teacher_preds,T):
    #T = 1
    alpha = 0.9
    loss = F.kl_div(F.log_softmax(preds / T, dim=1), F.softmax(teacher_preds / T, dim=1),
                    reduction='batchmean') * T * T * alpha + F.cross_entropy(preds, labels) * (1. - alpha)
    return loss


def train_one_epoch(model: torch.nn.Module,model_convxt: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,optimizer_t: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, mask=None,T=1):
    model.train()
    model_convxt.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    loss_ce = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq

        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output_t = model(samples)
                output=model_convxt(samples)
                #loss = criterion(output, targets)
                #loss=loss_kd(output,targets,output_t,T)
                loss_s=loss_ce(output,targets)+loss_kl(F.log_softmax(output,dim=1),F.softmax(output_t.detach(),dim=1))
                loss_t=loss_ce(output_t,targets)+loss_kl(F.log_softmax(output_t,dim=1),F.softmax(output.detach(),dim=1))
        else: # full precision
            output_t = model(samples)
            output=model_convxt(samples)
            #loss = criterion(output, targets)
            #loss=loss_kd(output,targets,output_t,T)
            loss_s=loss_ce(output,targets)+loss_kl(F.log_softmax(output,dim=1),F.softmax(output_t.detach(),dim=1))
            loss_t=loss_ce(output_t,targets)+loss_kl(F.log_softmax(output_t,dim=1),F.softmax(output.detach(),dim=1))

        loss_value = (loss_s+loss_t).item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_s /= update_freq
            loss_t /= update_freq
            grad_norm_t = loss_scaler(loss_t, optimizer_t, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            grad_norm_s = loss_scaler(loss_s, optimizer, clip_grad=max_norm,
                                    parameters=model_convxt.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)

                
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                optimizer_t.zero_grad()
                if model_ema is not None:
                    model_ema.update(model_convxt, mask)

        else: # full precision
            loss_s /= update_freq
            loss_s.backward()
            loss_t/=update_freq
            loss_t.backward()

            if (data_iter_step + 1) % update_freq == 0:
                if mask:
                    mask.step()
                    optimizer.step()
                else:
                    optimizer.step()
                    optimizer_t.step()
                    
                optimizer.zero_grad()
                optimizer_t.zero_grad()
                if model_ema is not None:
                    model_ema.update(model_convxt, mask)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc_s = (output.max(-1)[-1] == targets).float().mean()
            class_acc_t = (output_t.max(-1)[-1] == targets).float().mean()
        else:
            class_acc_s = None
            class_acc_t=None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc_s=class_acc_s)
        metric_logger.update(class_acc_t=class_acc_t)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        for group in optimizer_t.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm_t=grad_norm_t)
            metric_logger.update(grad_norm_s=grad_norm_s)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc_s=class_acc_s, head="loss")
            log_writer.update(class_acc_t=class_acc_t, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm_s=grad_norm_s, head="opt")
                log_writer.update(grad_norm_t=grad_norm_t, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc_s:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc_s': class_acc_s}, commit=False)
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc_t': class_acc_t}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm_s': grad_norm_s}, commit=False)
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm_t': grad_norm_t}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
