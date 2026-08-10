import torch.nn as nn


class EmulatorNN(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.GELU(),

            nn.Linear(256, 256),
            nn.GELU(),

            nn.Linear(256, 256),
            nn.GELU(),

            nn.Linear(256, output_dim)

        )

    def forward(self, x):

        return self.net(x)
