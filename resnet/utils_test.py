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
    "dropout_rate", "num_classes", "image_size"])

# Parameters for an individual model block
BlockArgs = collections.namedtuple("BlockArgs", [
    "kernel_size", "stride", "padding"])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def resnet_params(model_name):
    """ Map resnet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:  res,dropout
        "resnet20":     (32, 0.2),
        "resnet32":     (32, 0.2),
        "resnet44":     (32, 0.2),
        "resnet56":     (32, 0.2),
        "resnet110":    (32, 0.2),
        "resnet1202":   (32, 0.2),
    }
    return params_dict[model_name]


def resnet(model_name, dropout_rate=0.2, image_size=None, num_classes=10):
    """ Creates a resnet model. """

    layers_dict = {
        # Coefficients:  block layers list
        "resnet20": [3, 3, 3],
        "resnet32": [5, 5, 5],
        "resnet44": [7, 7, 7],
        "resnet56": [9, 9, 9],
        "resnet110": [18, 18, 18],
        "resnet1202": [200, 200, 200],
    }
    block_layers = layers_dict[model_name]

    global_params = GlobalParams(
        dropout_rate=dropout_rate,
        image_size=image_size,
        num_classes=num_classes,
    )

    return block_layers, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith("resnet"):
        s, p = resnet_params(model_name)
        blocks_args, global_params = resnet(
            model_name=model_name, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError("model name is not pre-defined: %s" % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


url_map = {
    "resnet20": "https://github.com/Lornatang/models/raw/master/resnet/resnet20-546fab9e.pth",
    "resnet32": "https://github.com/Lornatang/models/raw/master/resnet/resnet32-b9948351.pth",
    "resnet44": "https://github.com/Lornatang/models/raw/master/resnet/resnet44-f74dd615.pth",
    "resnet56": "https://github.com/Lornatang/models/raw/master/resnet/resnet56-68aecbac.pth",
    "resnet110": "https://github.com/Lornatang/models/raw/master/resnet/resnet110-000407b3.pth",
}


def load_pretrained_weights(model, model_name, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(url_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(["fc.weight", "fc.bias"]), "issue loading pretrained weights"
    print("Loaded pretrained weights for {}".format(model_name))
