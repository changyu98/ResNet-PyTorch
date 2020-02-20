# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import collections
import torch
from torch.utils import model_zoo

########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple("GlobalParams", [
    "block", "layers", "zero_init_residual",
    "groups", "width_per_group", "replace_stride_with_dilation",
    "norm_layer", "num_classes", "image_size"])

# Parameters for an individual model block
BlockArgs = collections.namedtuple("BlockArgs", [
    "layers"])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def resnet_params(model_name):
    """ Map resnet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   block, res
        "resnet18":  ("BasicBlock", 224),
        "resnet30":  ("BasicBlock", 224),
        "resnet54":  ("Bottleneck", 224),
        "resnet101": ("Bottleneck", 224),
        "resnet152": ("Bottleneck", 224),
    }
    return params_dict[model_name]


def resnet(model_name, block, num_classes=1000, zero_init_residual=False,
           groups=1, width_per_group=64, replace_stride_with_dilation=None,
           norm_layer=None, image_size=224):
    """ Creates a resnet model. """

    blocks_dict = {
        "resnet18":  (2, 2, 2, 2),
        "resnet30":  (3, 4, 6, 3),
        "resnet54":  (3, 4, 6, 3),
        "resnet101": (3, 4, 23, 3),
        "resnet152": (3, 8, 36, 3),
    }

    blocks_args = blocks_dict[model_name]

    global_params = GlobalParams(
        block=block,
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
        groups=groups,
        width_per_group=width_per_group,
        replace_stride_with_dilation=replace_stride_with_dilation,
        norm_layer=norm_layer,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith("resnet"):
        b, s = resnet_params(model_name)
        blocks_args, global_params = resnet(
            model_name=model_name, block=b, image_size=s)
    else:
        raise NotImplementedError("model name is not pre-defined: %s" % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return list(blocks_args), global_params


urls_map = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def load_pretrained_weights(model, model_name, load_fc=True):
    # "."s are no longer allowed in module names, but previous DenseLayer
    # has keys "norm.1", "relu.1", "conv.1", "norm.2", "relu.2", "conv.2".
    # They are also in the checkpoints in urls_map. This pattern is used
    # to find such keys.
    state_dict = model_zoo.load_url(urls_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights for {model_name}.")
