from datetime import datetime
from typing import Iterator, List, Tuple

import torch as T
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms  # type: ignore

from model import AlexNet, MiniAlexNet

ROOT_DATA_DIR = ".data"


def get_mnist(batch_size: int) -> Tuple[DataLoader, DataLoader, List[int], int]:
    input_dims = [28, 28]
    n_classes = 10

    mnist_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.0, 0.5),
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

    return training_data_loader, validation_data_loader, input_dims, n_classes


def get_imagenet(
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, List[int], int]:
    input_dims = [224, 224]
    n_classes = 1000

    training_set = datasets.ImageNet(ROOT_DATA_DIR, train=True)
    validation_set = datasets.ImageNet(ROOT_DATA_DIR, train=False)

    training_data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_set, batch_size=1000)

    print(training_set)
    print(validation_set)

    return training_data_loader, validation_data_loader, input_dims, n_classes


def train(
    data_loader: DataLoader, model: nn.Module, optimizer: Optimizer, n_epochs: int
) -> None:
    for epoch in range(n_epochs):
        epoch_loss = T.scalar_tensor(0.0)
        for inp, target in data_loader:
            optimizer.zero_grad()
            output = model(inp)
            loss = F.cross_entropy(output, target)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        predictions = T.argmax(output, dim=-1)
        n_batches = len(data_loader)
        correct = T.eq(predictions, target)
        n_correct = correct.sum()
        batch_size = data_loader.batch_size
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] Epoch {epoch}; Loss {epoch_loss/n_batches}; Correct: {n_correct/batch_size:.2%}"
        )


def evaluate(data_loader: Iterator, model: nn.Module) -> None:
    inp, target = next(data_loader)
    n_examples = inp.size()[0]
    output = model(inp)
    predictions = T.argmax(output, dim=-1)
    correct = T.eq(predictions, target)
    n_correct = correct.sum()
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] Validation correct: {n_correct/n_examples:.2%}"
    )


def main() -> None:
    n_epochs = 10
    batch_size = 128

    training_data_loader, validation_data_loader, input_dims, n_classes = get_mnist(
        batch_size
    )
    # training_data_loader, validation_data_loader, input_dims, n_classes = get_imagenet(batch_size)
    validation_data_iterator = iter(validation_data_loader)

    # model = AlexNet(input_dims, n_classes)
    model = MiniAlexNet(input_dims, n_classes)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    print("Pre-training evaluation...")
    evaluate(validation_data_iterator, model)

    print("Training...")
    train(training_data_loader, model, optimizer, n_epochs=n_epochs)

    print("Post-training evaluation...")
    evaluate(validation_data_iterator, model)


main()
