import timm
from torch import nn


def make_model():
    return BeitTransformer()


class BeitTransformer(nn.Module):
    def __init__(self):
        super(BeitTransformer, self).__init__()
        self.model = timm.create_model("beit_large_patch16_224")
        self.model.head = nn.Linear(1024, 2)

    def forward(self, inputs):
        output = self.model(inputs)
        return output
