from cnn.vgg_block import VGGBlock
from torch.nn import (
    Module,
    Flatten,
    Linear
)

class CNN(Module):
    """CNN Model for cats and dogs classification"""

    def __init__(self):
        super().__init__()
        self.vgg1 = VGGBlock(3, 8, 3, 2)
        self.vgg2 = VGGBlock(8, 16, 3, 2)
        self.vgg3 = VGGBlock(16, 32, 3, 2)
        self.vgg4 = VGGBlock(32, 64, 3, 2)
        self.vgg5 = VGGBlock(64, 128, 3, 2)
        self.flatten = Flatten()
        self.output = Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = self.vgg1(x)
        x = self.vgg2(x)
        x = self.vgg3(x)
        x = self.vgg4(x)
        x = self.vgg5(x)
        x = self.flatten(x)
        x = self.output(x)
        return x