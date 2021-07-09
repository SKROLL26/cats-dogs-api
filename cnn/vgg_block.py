from torch.nn import (
    Module,
    ZeroPad2d,
    Conv2d,
    MaxPool2d,
    ReLU,
    BatchNorm2d
)

class VGGBlock(Module):
    """VGG Block module"""

    def __init__(self, in_channels, out_channels, conv_kernel_size, maxpool_kernel_size):
        super().__init__()
        self.zero_pad = ZeroPad2d(1)
        self.conv2d = Conv2d(in_channels, out_channels, conv_kernel_size)
        self.maxpool2d = MaxPool2d(maxpool_kernel_size)
        self.relu = ReLU()
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.zero_pad(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.maxpool2d(x)
        return x

