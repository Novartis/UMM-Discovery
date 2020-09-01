# Copyright 2020 Novartis Institutes for BioMedical Research Inc.
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

import torch
import torch.nn as nn
import math
from string import ascii_lowercase

class MultiScaleNet(nn.Module):
    def __init__(self, input_dim, num_features, num_classes):
        super(MultiScaleNet, self).__init__()
        self.features = make_layers(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128, num_features),
            nn.ReLU(True),
        )
        self.top_layer = nn.Linear(num_features, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                # print(y)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(input_dim):
    layers = []
    layers.append(MSBlock(input_dim, 6, 9))
    layers.append(MSBlock(12, 12, 20))
    layers.append(MSBlock(32, 16, 32))
    layers.append(MSBlock(64, 16, 32))
    layers.append(MSBlock(96, 16, 32))
    layers.append(Collapse(128))
    layers.append(Dense(128)),
    layers.append(nn.BatchNorm2d(128)),
    layers.append(nn.ReLU(True)),
    return nn.Sequential(*layers)


class MSBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(MSBlock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dense1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dense2 = nn.Conv2d(in_channels + out_channels, in_channels + out_channels, kernel_size=1, padding=0)
        self.bn4 = nn.BatchNorm2d(in_channels + out_channels)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.pool1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dense1(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool2(out)
        out = torch.cat((y, out), 1)
        out = self.dense2(out)
        out = self.bn4(out)
        out = self.relu4(out)
        return out


class Dense(nn.Module):
    def __init__(self, input_features, output_features=None):
        super(Dense, self).__init__()
        self.input_features = input_features
        self.output_features = input_features if output_features is None else output_features
        self.weight = nn.Parameter(torch.Tensor(input_features, self.output_features), requires_grad=True)
        self.weight.data.normal_(0, math.sqrt(2. / input_features))
        self.register_parameter('bias', None)

    def forward(self, x):
        return self.dense(x)

    def dense(self, inputs):
        eqn = 'ay{0},yz->az{0}'.format(ascii_lowercase[1:3])
        return torch.einsum(eqn, inputs, self.weight)


class Collapse(nn.Module):
    def __init__(self, size):
        super(Collapse, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(size), requires_grad=True)
        self.weight.data.zero_()
        self.p_avg_l = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.p_max_l = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        return self.collapse(x)

    def collapse(self, inputs):
        p_avg = self.p_avg_l(inputs)
        p_max = self.p_max_l(inputs)

        factor = torch.sigmoid(self.weight)
        eqn = 'ay{0},y->ay{0}'.format(ascii_lowercase[1:3])
        return torch.einsum(eqn, [p_avg, factor]) + torch.einsum(eqn, [p_max, torch.sub(1.0, factor)])
