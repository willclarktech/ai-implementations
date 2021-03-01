from typing import List

import torch as T
from torch import nn


class AlexNet(nn.Module):
    """
    Based on the original architecture targeting the ImageNet dataset
    """

    def __init__(self, input_dims: List[int], n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 3),
            nn.ReLU(),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Flatten(),
            nn.Linear(256, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, n_classes),
            nn.ReLU(),
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)


class MiniAlexNet(nn.Module):
    """
    A simplified version of AlexNet targeting the MNIST dataset
    """

    def __init__(self, input_dims: List[int], n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.ReLU(),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Flatten(),
            nn.Linear(576, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, n_classes),
            nn.ReLU(),
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.net(x)
