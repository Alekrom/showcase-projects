# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=wildcard-import, arguments-differ
r"""Module for pre-defined neural network models.

This module contains definitions for the following model architectures:
-  `AlexNet`_
-  `DenseNet`_
-  `Inception V3`_
-  `ResNet V1`_
-  `ResNet V2`_
-  `SqueezeNet`_
-  `VGG`_
-  `MobileNet`_
-  `MobileNetV2`_

You can construct a model with random weights by calling its constructor:

.. code::

    from mxnet.gluon.model_zoo import vision
    resnet18 = vision.resnet18_v1()
    alexnet = vision.alexnet()
    squeezenet = vision.squeezenet1_0()
    densenet = vision.densenet_161()

We provide pre-trained models for all the listed models.
These models can constructed by passing ``pretrained=True``:

.. code::

    from mxnet.gluon.model_zoo import vision
    resnet18 = vision.resnet18_v1(pretrained=True)
    alexnet = vision.alexnet(pretrained=True)

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape (N x 3 x H x W),
where N is the batch size, and H and W are expected to be at least 224.
The images have to be loaded in to a range of [0, 1] and then normalized
using ``mean = [0.485, 0.456, 0.406]`` and ``std = [0.229, 0.224, 0.225]``.
The transformation should preferrably happen at preprocessing. You can use
``mx.image.color_normalize`` for such transformation::

    image = image/255
    normalized = mx.image.color_normalize(image,
                                          mean=mx.nd.array([0.485, 0.456, 0.406]),
                                          std=mx.nd.array([0.229, 0.224, 0.225]))

.. _AlexNet: https://arxiv.org/abs/1404.5997
.. _DenseNet: https://arxiv.org/abs/1608.06993
.. _Inception V3: http://arxiv.org/abs/1512.00567
.. _ResNet V1: https://arxiv.org/abs/1512.03385
.. _ResNet V2: https://arxiv.org/abs/1603.05027
.. _SqueezeNet: https://arxiv.org/abs/1602.07360
.. _VGG: https://arxiv.org/abs/1409.1556
.. _MobileNet: https://arxiv.org/abs/1704.04861
.. _MobileNetV2: https://arxiv.org/abs/1801.04381
"""

from .resnet import *
from .basenet_dense import *
from .resnet_e import *

def get_model_parameters():
    return [ResNetEParameters()]


def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    models = {
        'resnet18_v2':  resnet18_v2,
        'resnet18_e1': resnet18_e1,
        'resnet34_e1': resnet34_e1,
        'resnet50_e1': resnet50_e1,
        'resnet101_e1': resnet101_e1,
        'resnet152_e1': resnet152_e1,
        'resnet18_e2': resnet18_e2,
        'resnet34_e2': resnet34_e2,
        'resnet50_e2': resnet50_e2,
        'resnet101_e2': resnet101_e2,
        'resnet152_e2': resnet152_e2,
    }
    name = name.lower()
    name = name
    if name not in models:
        raise ValueError(
            'Model %s is not supported. Available options are\n\t%s' % (
                name, '\n\t'.join(sorted(models.keys()))))
    return models[name](**kwargs)
