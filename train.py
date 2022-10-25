# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train"""
import math
import os

import numpy as np
from mindspore import Model
from mindspore import Tensor
from mindspore import context
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.common import set_seed
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn import LossBase
from mindspore.nn.optim import AdamWeightDecay
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor

from src.args import args
from src.beit import beit_base_patch16_224
from src.callback import EvaluateCallBack
from src.imagenet import create_dataset_imagenet
from src.moxing_adapter import sync_data
from src.utils import pretrained


def main():
    set_seed(args.seed)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)

    # set context and device
    device_target = args.device_target
    device_num = int(os.environ.get("DEVICE_NUM", 1))
    rank = 0
    if device_target == "Ascend":
        if device_num > 1:
            context.set_context(device_id=int(os.environ["DEVICE_ID"]))
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    else:
        raise ValueError("Unsupported platform.")

    # get model and cast amp_level
    model = beit_base_patch16_224(
        drop_path_rate=0.1,
        num_classes=args.num_classes,
        use_rel_pos_bias=True,
        use_abs_pos_emb=False,
        init_values=0.1
    )
    cell_types = (nn.LayerNorm, nn.Softmax, nn.BatchNorm2d, nn.GELU, nn.SyncBatchNorm, nn.GroupNorm)
    model.to_float(mstype.float16)
    do_keep_fp32(model, cell_types)
    loss_fn = CrossEntropySmooth(sparse=True, num_classes=args.num_classes, smooth_factor=args.label_smoothing)
    net_with_loss = nn.WithLossCell(model, loss_fn)

    eval_network = nn.WithEvalCell(network=model, loss_fn=loss_fn)

    pretrained(args, model)
    dataset_train_url = os.path.join(args.data_url, "train")
    dataset_val_url = os.path.join(args.data_url, "val")


    if args.run_modelarts:
        sync_data(args.data_url, "/cache/dataset", threads=128)
        dataset_train_url = "/cache/dataset/imagenet/train"
        dataset_val_url = "/cache/dataset/imagenet/val"
    train_dataset = create_dataset_imagenet(dataset_train_url, training=True, args=args)
    val_dataset = create_dataset_imagenet(dataset_val_url, training=False, args=args)
    batch_num = train_dataset.get_dataset_size()

    learning_rate = warmup_cosine_annealing_lr(args.base_lr, batch_num, args.warmup_length, args.epochs)
    params = get_param_groups(model)
    optimizer = AdamWeightDecay(
        params=params,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay
    )

    scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 16, scale_factor=2,
                                                                scale_window=2000)
    train_one_steo = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=scale_sense)

    model = Model(train_one_steo, metrics={"acc1": nn.Top1CategoricalAccuracy(), "loss": nn.Loss()},
                  eval_network=eval_network, eval_indexes=[0, 1, 2])

    ckpt_save_dir = "./ckpt_" + str(rank)
    if args.run_modelarts:
        ckpt_save_dir = "/cache/ckpt_" + str(rank)
    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num,
                                 keep_checkpoint_max=args.save_every)
    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    ckpoint_cb = ModelCheckpoint(prefix="BEiT", directory=ckpt_save_dir, config=config_ck)
    loss_cb = LossMonitor(per_print_times=200)
    eval_cb = EvaluateCallBack(model, eval_dataset=val_dataset, src_url=ckpt_save_dir,
                               train_url=os.path.join(args.train_url, "ckpt_" + str(rank)))

    model.train(args.epochs, train_dataset, callbacks=[time_cb, ckpoint_cb, loss_cb, eval_cb])


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)
        self.cast = ops.Cast()

    def construct(self, logit, label):
        logit = ops.Cast()(logit, mstype.float32)
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        label = ops.Cast()(label, mstype.float32)
        loss2 = self.ce(logit, label)
        return loss2


def get_param_groups(network):
    """ get param groups. """
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)
    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr1 = float(init_lr) + lr_inc * current_step
    return lr1


def warmup_cosine_annealing_lr(lr5, steps_per_epoch, warmup_epochs, max_epoch, eta_min=0):
    """ warmup cosine annealing lr."""
    base_lr = lr5
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr5 = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr5 = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / max_epoch)) / 2
        lr_each_step.append(lr5)

    return np.array(lr_each_step).astype(np.float32)


def do_keep_fp32(network, cell_types):
    """Cast cell to fp32 if cell in cell_types"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, cell_types):
            cell.to_float(mstype.float32)


if __name__ == '__main__':
    main()
