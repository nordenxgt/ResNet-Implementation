from typing import Type, Union, List

import torch
from torch import nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: int = 1, 
        stride: int = 1, 
        padding: int = 0
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), 
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.res_block1 = Conv(in_channels, out_channels, 3, stride, 1)
        self.res_block2 = Conv(out_channels, out_channels, 3, 1, 1)
        if stride != 1 or in_channels != self.expansion*out_channels: 
            self.shortcut = Conv(in_channels, self.expansion*out_channels, stride=stride)  
        else: 
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.res_block1(x))
        out = F.relu(self.res_block2(out) + self.shortcut(x))
        return out

class BottleneckBlock(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.bottleneck_block1 = Conv(in_channels, out_channels)
        self.bottleneck_block2 = Conv(out_channels, out_channels, 3, stride, 1)
        self.bottleneck_block3 = Conv(out_channels, self.expansion*out_channels)
        
        if stride != 1 or in_channels != self.expansion*out_channels: 
            self.shortcut = Conv(in_channels, self.expansion*out_channels, stride=stride)  
        else: 
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bottleneck_block1(x))
        out = F.relu(self.bottleneck_block2(out))
        out = F.relu(self.bottleneck_block3(out) + self.shortcut(x))
        return out
    
class ResNet(nn.Module):
    def __init__(self, num_layers: int, num_classes: int = 1000) -> None:
        super().__init__()
        block_config = {
            18: {"Block": ResBlock, "num_blocks": [2, 2, 2, 2]},
            34: {"Block": ResBlock, "num_blocks": [3, 4, 6, 3]},
            50: {"Block": BottleneckBlock, "num_blocks": [3, 4, 6, 3]},
            101: {"Block": BottleneckBlock, "num_blocks": [3, 4, 23, 3]},
            152: {"Block": BottleneckBlock, "num_blocks": [3, 8, 36, 3]}
        }

        self.block = block_config[num_layers]["Block"]
        self.num_blocks = block_config[num_layers]["num_blocks"]
        self.channels = 64

        self.conv1 = Conv(3, self.channels, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.conv2_x = self._make_layers(self.block, self.num_blocks[0], 64, stride=1)
        self.conv3_x = self._make_layers(self.block, self.num_blocks[1], 128, stride=2)
        self.conv4_x = self._make_layers(self.block, self.num_blocks[2], 256, stride=2)
        self.conv5_x = self._make_layers(self.block, self.num_blocks[3], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.block.expansion, num_classes)

        self._init_layers()

    def _make_layers(
        self, 
        Block: Type[Union[ResBlock, BottleneckBlock]],
        num_blocks: List[int],
        out_channels: int,
        stride: int
    ) -> nn.Sequential:
        layers = []
        layers.append(Block(self.channels, out_channels, stride))
        self.channels = Block.expansion*out_channels
        for _ in range(1, num_blocks):
            layers.append(Block(self.channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _init_layers(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
