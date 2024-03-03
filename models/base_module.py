import torch
import torch.nn as nn

class BaseNNModule(nn.Module):
    def __init__(self):
        super(BaseNNModule, self).__init__()

    def init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def normalize_input(self, x):
        return x.float() / 255.0
