# Copyright © 2023 Huawei Technologies Co, Ltd. All Rights Reserved.
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


import numpy as np
import pytest
import torch
import torch.nn as nn

from tunable_conv import TunableConv2d, TunableConvTranspose2d, TunableParameter

torch.manual_seed(1)
torch.random.manual_seed(1)
np.random.seed(1)


@pytest.mark.parametrize("num_params", [2, 3])
@pytest.mark.parametrize("default_input", [torch.empty((1, 1, 1)).uniform_()])
def test_tunable_parameter(default_input, num_params):
    assert num_params > 1
    batch_size = num_params
    gamma = TunableParameter(
        data=default_input,
        requires_grad=True,
        num_params=num_params,
        mode="linear",
    )
    px = torch.eye(num_params, num_params)
    g = gamma(px)
    print(gamma)
    assert g.shape == (batch_size, *default_input.shape)


@pytest.mark.parametrize("num_params", [3])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("groups", [1])
def test_tunable_conv2d(num_params, kernel_size, stride, groups):
    assert num_params > 1
    mse = nn.MSELoss()
    batch_size = num_params
    b, c, h, w, d = batch_size, 16, 24, 24, 32
    x = torch.empty((b, c, h, w)).normal_()
    px = torch.eye(num_params, num_params)

    tunable_conv = TunableConv2d(
        c,
        d,
        kernel_size,
        stride=stride,
        groups=groups,
        bias=True,
        num_params=num_params,
        mode="linear",
    )
    conv = nn.Conv2d(c, d, kernel_size, stride=stride, groups=groups, bias=True)
    print(tunable_conv)
    y = tunable_conv(x, px)
    for p in range(num_params):
        conv.weight.data = tunable_conv.weight[0, p, ...]
        conv.bias.data = tunable_conv.bias[0, p, ...]
        y_p = conv(x[p : p + 1, ...])

        assert mse(y[p : p + 1, ...], y_p) < 1e-6


@pytest.mark.parametrize("num_params", [3])
@pytest.mark.parametrize("kernel_size", [3])
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("groups", [1])
def test_tunable_convt2d(num_params, kernel_size, stride, groups):
    assert num_params > 1
    mse = nn.MSELoss()
    batch_size = num_params
    b, c, h, w, d = batch_size, 16, 24, 24, 32
    x = torch.empty((b, c, h, w)).normal_()
    px = torch.eye(num_params, num_params)

    tunable_convt = TunableConvTranspose2d(
        c,
        d,
        kernel_size,
        stride=stride,
        groups=groups,
        padding=1,
        output_padding=1,
        bias=True,
        num_params=num_params,
        mode="linear",
    )
    convt = nn.ConvTranspose2d(
        c,
        d,
        kernel_size,
        stride=stride,
        padding=1,
        output_padding=1,
        groups=groups,
        bias=True,
    )
    print(tunable_convt)
    y = tunable_convt(x, px)
    for p in range(num_params):
        convt.weight.data = tunable_convt.weight[0, p, ...]
        convt.bias.data = tunable_convt.bias[0, p, ...]
        y_p = convt(x[p : p + 1, ...])

        assert mse(y[p : p + 1, ...], y_p) < 1e-6
