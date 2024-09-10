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

# TunableBlock and TunableSequential abstractions were copied from
# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py

import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class TunableBlock(nn.Module):
    """
    Any module where forward() takes tunable parameters as a second argument.
    """

    @abstractmethod
    def forward(self, x, px):
        """
        Apply the module to `x` given `px` tunable parameters.
        """


class TunableSequential(nn.Sequential, TunableBlock):
    """
    A sequential module that passes tunable parameters to the children that
    support it as an extra input.
    """

    def forward(self, x, px):
        for layer in self:
            if isinstance(layer, TunableBlock):
                x = layer(x, px)
            else:
                x = layer(x)
        return x


class TunableModule(TunableBlock):
    def __init__(self, num_params=2, expand_params=1, mode="mlp"):
        super().__init__()

        self.num_params = num_params
        self.expand_params = expand_params or 1
        self.mode = mode or "mlp"

        self.num_weights = self.num_params * self.expand_params

        assert self.mode in ["linear", "mlp"]

        self.mlp = None
        if self.expand_params > 1 or self.mode == "mlp":
            self.mlp = nn.Linear(
                self.num_params,
                self.num_weights,
                bias=True,
            )
            self.init_mlp_parameters()
        self.weight = None
        self.bias = None

    def init_mlp_parameters(self):
        nn.Linear.reset_parameters(self.mlp)

    def _init_parameter(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "num_params={}, expand_params={}, mode={}".format(
            self.num_params, self.expand_params, self.mode
        )
        return s

    def check_input(self, px: torch.Tensor):
        assert px is not None
        assert px.ndim == 2, px.shape
        assert px.shape[1] == self.num_params, (px.shape, self.num_params)
        return 0


class TunableParameter(TunableModule):
    def __init__(
        self,
        data: torch.Tensor,
        requires_grad=True,
        num_params=2,
        expand_params=1,
        mode="mlp",
    ):
        super().__init__(num_params=num_params, expand_params=expand_params, mode=mode)

        self.requires_grad = requires_grad
        self.ndim = data.ndim
        data = torch.repeat_interleave(
            data.reshape(1, 1, -1), repeats=self.num_weights, dim=1
        ).reshape(1, self.num_weights, *data.shape)
        self.data = nn.Parameter(data=data, requires_grad=requires_grad)

    def extra_repr(self):
        s = "shape={}, dtype={}, requires_grad={}".format(
            self.data.shape, self.data.dtype, self.requires_grad
        )
        s += ", {}".format(TunableModule.extra_repr(self))
        return s

    def forward(self, px: torch.Tensor):
        self.check_input(px)
        b = px.shape[0]
        w = self.num_weights

        if self.mlp is not None:
            px = self.mlp(px)

        px = px.reshape(b, w, *[1 for _ in range(self.ndim)])
        data = (px * self.data).sum(axis=1)

        return data


class TunableLinear(TunableModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_params: int = 1,
        expand_params: int = 1,
        mode: str = "mlp",
    ):
        super().__init__(num_params=num_params, expand_params=expand_params, mode=mode)

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty([1, self.num_weights, out_features, in_features]),
            requires_grad=True,
        )
        self.bias = (
            nn.Parameter(
                torch.empty([1, self.num_weights, out_features], requires_grad=True)
            )
            if bias
            else None
        )
        self._init_parameter()

    def extra_repr(self):
        s = "in_features={}, out_features={}".format(
            self.in_features, self.out_features
        )
        if self.bias:
            s += ", bias={}".format(self.bias)
        s += ", {}".format(TunableModule.extra_repr(self))
        return s

    def forward(self, x: torch.Tensor, px: torch.Tensor):
        self.check_input(px)

        b = x.shape[0]
        w = self.num_weights
        c = self.in_features
        d = self.out_features

        if self.mlp is not None:
            px = self.mlp(px)

        weight = (px.reshape(b, w, 1, 1) * self.weight).sum(dim=1, keepdim=False)
        bias = (
            (px.reshape(b, w, 1) * self.bias).sum(dim=1, keepdim=True)
            if self.bias is not None
            else None
        )

        assert weight.shape == (b, d, c), weight.shape
        assert bias.shape == (b, 1, d), bias.shape

        y = x.view(b, -1, c) @ weight + bias

        return y


class TunableConv2d(TunableModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        num_params=1,
        expand_params=1,
        mode="mlp",
    ):
        super().__init__(num_params=num_params, expand_params=expand_params, mode=mode)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        assert in_channels % self.groups == 0

        self.weight = nn.Parameter(
            torch.empty(
                [
                    1,
                    self.num_weights,
                    out_channels,
                    in_channels // self.groups,
                    kernel_size,
                    kernel_size,
                ]
            ),
            requires_grad=True,
        )
        self.bias = (
            nn.Parameter(
                torch.empty([1, self.num_weights, out_channels]), requires_grad=True
            )
            if bias
            else None
        )
        self._init_parameter()

    def extra_repr(self):
        s = (
            "input_channels={}, output_channels={}, kernel_size={},"
            "stride={}, padding={}, dilation={}, "
            "groups={}, bias={}"
            ", {}".format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.bias,
                TunableModule.extra_repr(self),
            )
        )
        return s

    def _batch_conv2d(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
    ):
        assert weight.shape[1] % self.groups == 0, weight.shape
        assert x.ndim == 4 and weight.ndim == 5
        assert x.shape[0] == weight.shape[0]

        b, in_ch, h, w = x.shape
        _, out_ch, _, kh, kw = weight.shape
        y = F.conv2d(
            x.reshape(1, b * in_ch, h, w),
            weight.reshape(b * out_ch, in_ch // self.groups, kh, kw),
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=b * self.groups,
        )
        y = y.reshape(b, out_ch, y.shape[2], y.shape[3])

        if bias is not None:
            assert bias.ndim == 2
            assert b == bias.shape[0] and out_ch == bias.shape[1]
            y = y + bias.reshape(b, out_ch, 1, 1)

        return y

    def forward(self, x: torch.Tensor, px: torch.Tensor):
        self.check_input(px)

        b = x.shape[0]
        w = self.num_weights

        if self.mlp is not None:
            px = self.mlp(px)

        weight = (px.view(b, w, 1, 1, 1, 1) * self.weight).sum(axis=1)
        bias = (
            (px.view(b, w, 1) * self.bias).sum(axis=1)
            if self.bias is not None
            else None
        )

        y = self._batch_conv2d(x, weight, bias=bias)
        return y


# NOTE Experimental code, not introduced in the paper


class TunableConvTranspose2d(TunableModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=False,
        dilation=1,
        num_params=1,
        expand_params=1,
        mode="mlp",
    ):
        super().__init__(num_params=num_params, expand_params=expand_params, mode=mode)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.output_padding = output_padding
        assert in_channels % self.groups == 0

        self.weight = nn.Parameter(
            torch.empty(
                [
                    1,
                    self.num_weights,
                    in_channels // self.groups,
                    out_channels,
                    kernel_size,
                    kernel_size,
                ]
            ),
            requires_grad=True,
        )
        self.bias = (
            nn.Parameter(
                torch.empty([1, self.num_weights, out_channels]), requires_grad=True
            )
            if bias
            else None
        )
        self._init_parameter()

    def extra_repr(self):
        s = (
            "input_channels={}, output_channels={}, kernel_size={},"
            "stride={}, output_padding={}, padding={}, dilation={}, "
            "groups={}, bias={}"
            ", {}".format(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.output_padding,
                self.padding,
                self.dilation,
                self.groups,
                self.bias,
                TunableModule.extra_repr(self),
            )
        )
        return s

    def _batch_convt2d(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
    ):
        assert weight.shape[1] % self.groups == 0, weight.shape
        assert x.ndim == 4 and weight.ndim == 5
        assert x.shape[0] == weight.shape[0]

        b, in_ch, h, w = x.shape
        _, _, out_ch, kh, kw = weight.shape
        output_padding = nn.ConvTranspose2d._output_padding(
            self,
            x,
            None,
            self.stride,
            self.padding,
            self.kernel_size,
            2,
            self.dilation,
        )
        y = F.conv_transpose2d(
            x.reshape(1, b * in_ch, h, w),
            weight.reshape(b * in_ch // self.groups, out_ch, kh, kw),
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
            dilation=self.dilation,
            groups=b * self.groups,
        )
        y = y.reshape(b, out_ch, y.shape[2], y.shape[3])

        if bias is not None:
            assert bias.ndim == 2
            assert b == bias.shape[0] and out_ch == bias.shape[1]
            y = y + bias.reshape(b, out_ch, 1, 1)

        return y

    def forward(self, x: torch.Tensor, px: torch.Tensor):
        self.check_input(px)

        b = x.shape[0]
        w = self.num_weights

        if self.mlp is not None:
            px = self.mlp(px)

        weight = (px.view(b, w, 1, 1, 1, 1) * self.weight).sum(axis=1)
        bias = (
            (px.view(b, w, 1) * self.bias).sum(axis=1)
            if self.bias is not None
            else None
        )
        y = self._batch_convt2d(x, weight, bias=bias)
        return y
