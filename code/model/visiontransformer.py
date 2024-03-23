import timm
from torch import nn


def make_model():
    return Vittransformer()


class Vittransformer(nn.Module):
    def __init__(self):
        super(Vittransformer, self).__init__()
        self.model = timm.create_model("vit_base_patch8_224")
        self.model.head=nn.Linear(768, 2)

    def forward(self, inputs):
        output = self.model(inputs)
        return output
