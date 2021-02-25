from typing import Tuple

import torch as T
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms  # type: ignore

from model import AlexNet

ROOT_DATA_DIR = ".data"


def get_mnist(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    mnist_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    training_set = datasets.MNIST(
        ROOT_DATA_DIR, train=True, download=True, transform=mnist_transforms
    )
    validation_set = datasets.MNIST(
        ROOT_DATA_DIR, train=False, transform=mnist_transforms
    )
    training_data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_set, batch_size=1000)

    print(training_set)
    print(validation_set)

    return training_data_loader, validation_data_loader


def get_imagenet(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    training_set = datasets.ImageNet(ROOT_DATA_DIR, train=True)
    validation_set = datasets.ImageNet(ROOT_DATA_DIR, train=False)

    training_data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_set, batch_size=1000)

    print(training_set)
    print(validation_set)

    return training_data_loader, validation_data_loader


batch_size = 128
training_data_loader, validation_data_loader = get_imagenet(batch_size)


mnist_input_dims = [28, 28]
mnist_n_classes = 10
model = AlexNet(mnist_input_dims, mnist_n_classes)

optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)


def train(n_epochs: int):
    for epoch in range(n_epochs):
        epoch_loss = T.scalar_tensor(0.0)
        for i, (inp, target) in enumerate(training_data_loader):
            optimizer.zero_grad()
            output = model(inp)
            loss = F.cross_entropy(output, target)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

            if i > 19:
                break

        predictions = T.argmax(output, dim=-1)
        correct = T.eq(predictions, target)
        n_correct = correct.sum()
        print(
            f"Epoch {epoch}; Loss {epoch_loss/i}; Correct: {n_correct/batch_size:.2%}"
        )


def evaluate():
    for inp, target in validation_data_loader:
        n_examples = inp.size()[0]
        output = model(inp)
        predictions = T.argmax(output, dim=-1)
        correct = T.eq(predictions, target)
        n_correct = correct.sum()
        print(f"Validation correct: {n_correct/n_examples:.2%}")

        break


train(n_epochs=2)
evaluate()
