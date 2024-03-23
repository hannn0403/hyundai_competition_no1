import timm
from torch import nn


def make_model():
    return Swintrans()


class Swintrans(nn.Module):
    def __init__(self):
        super(Swintrans, self).__init__()
        self.model = timm.create_model("swin_large_patch4_window7_224")
        self.model.head = nn.Linear(1536, 2)

    def forward(self, inputs):
        output = self.model(inputs)
        return output
