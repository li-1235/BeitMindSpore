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
import argparse
import ast
import os
import sys

import yaml

from src.configs import parser as _parser

args = None


def parse_arguments():
    """parse_arguments"""
    global args
    parser = argparse.ArgumentParser(description="MindSpore BEiT Fitune Training")

    parser.add_argument("--amp_level", default="O2", choices=["O0", "O1", "O2", "O3"], help="AMP Level")
    parser.add_argument("--batch_size", default=256, type=int, metavar="N",
                        help="mini-batch size (default: 256), this is the total "
                             "batch size of all Devices on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--device_id", default=0, type=int, help="device id")
    parser.add_argument("--device_num", default=1, type=int, help="device num")
    parser.add_argument("--device_target", default="Ascend", choices=["GPU", "Ascend"], type=str)
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--is_dynamic_loss_scale", default=1, type=int, help="is_dynamic_loss_scale ")
    parser.add_argument("--keep_checkpoint_max", default=20, type=int, help="keep checkpoint max num")
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument("-j", "--num_parallel_workers", default=20, type=int, metavar="N",
                        help="number of data loading workers (default: 20)")
    parser.add_argument("--warmup_length", default=0, type=int, help="number of warmup iterations")
    parser.add_argument("--warmup_lr", default=5e-7, type=float, help="warm up learning rate")
    parser.add_argument("--wd", "--weight_decay", default=0.05, type=float, metavar="W",
                        help="weight decay (default: 0.05)", dest="weight_decay")
    parser.add_argument("--loss_scale", default=1024, type=int, help="loss_scale")
    parser.add_argument("--base_lr", default=5e-4, type=float, help="initial lr")
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--pretrained", dest="pretrained", default=None, type=str, help="use pre-trained model")
    parser.add_argument("--config", help="Config file to use (see configs dir)", default=None)
    parser.add_argument("--seed", default=0, type=int, help="seed for initializing training. ")
    parser.add_argument("--save_every", default=2, type=int, help="save every ___ epochs(default:2)")
    parser.add_argument("--label_smoothing", type=float, help="label smoothing to use, default 0.1", default=0.1)
    parser.add_argument('--data_url', default="./data", help='location of data.')
    parser.add_argument('--train_url', default="./", help='location of training outputs.')
    parser.add_argument("--run_modelarts", type=ast.literal_eval, default=False, help="whether run on modelarts")
    parser.add_argument('--multi_data_url', default="./data", help='location of data.')

    args = parser.parse_args()

    get_config()


def get_config():
    """get_config"""
    global args
    override_args = _parser.argv_to_vars(sys.argv)
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./configs",
        args.config,
    )
    yaml_txt = open(config_path).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)

    for v in override_args:
        loaded_yaml[v] = getattr(args, v)
    print(f"=> Reading YAML config from {config_path}")
    if args.run_modelarts:
        args.multi_data_url = eval(args.multi_data_url)
        if args.multi_data_url[0]['dataset_name'] == "imagenet":
            args.data_url = args.multi_data_url[0]['dataset_url']
            args.pretrained = args.multi_data_url[1]['dataset_url']
        else:
            args.data_url = args.multi_data_url[1]['dataset_url']
            args.pretrained = args.multi_data_url[0]['dataset_url']
    args.__dict__.update(loaded_yaml)
    print(args)
    os.environ["DEVICE_TARGET"] = args.device_target
    if "DEVICE_NUM" not in os.environ.keys():
        os.environ["DEVICE_NUM"] = str(args.device_num)
    if "RANK_SIZE" not in os.environ.keys():
        os.environ["RANK_SIZE"] = str(args.device_num)


def run_args():
    """run and get args"""
    global args
    if args is None:
        parse_arguments()


run_args()
