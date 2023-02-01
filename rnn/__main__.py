"""
Adapted from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""
from __future__ import division, print_function, unicode_literals

import glob
import math
import os
import random
import string
import time
import unicodedata
from io import open
from typing import Callable, Dict, List, Tuple, TypeVar, cast

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker as ticker  # type: ignore
import torch
from torch import nn, optim

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def time_since(since: float) -> str:
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def find_files(path: str) -> List[str]:
    return glob.glob(path)


def unicode_to_ascii(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != "Mn" and c in all_letters
    )


def read_lines(filename: str) -> List[str]:
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


def letter_to_index(letter: str) -> int:
    return all_letters.find(letter)


def letter_to_tensor(letter: str) -> torch.Tensor:
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line: str) -> torch.Tensor:
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


def category_from_output(output: torch.Tensor, categories: List[str]) -> Tuple[str, int]:
    _, top_i = output.topk(1)
    category_i = cast(int, top_i[0].item())
    return categories[category_i], category_i


T = TypeVar('T')
def random_choice(l: List[T]) -> T:
    return l[random.randint(0, len(l) - 1)]


def random_training_example(categories: List[str], category_lines: Dict[str, List[str]]) -> Tuple[str, str, torch.Tensor, torch.Tensor]:
    category = random_choice(categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def plot_losses(losses) -> None:
    plt.figure()
    plt.plot(losses)
    plt.show()


def plot_confusion(categories: List[str], confusion: torch.Tensor) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    ax.set_xticklabels([''] + categories, rotation=90)
    ax.set_yticklabels([''] + categories)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


class Rnn(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(Rnn, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)


    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden


    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, self.hidden_size)


    def train(self, category: torch.Tensor, line: torch.Tensor, optimizer: optim.Optimizer, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, float]:
        optimizer.zero_grad()
        hidden = self.init_hidden()

        for i in range(line.size()[0]):
            output, hidden = self(line[i], hidden)

        loss = criterion(output, category)
        loss.backward()
        optimizer.step()
        return output, loss.item()


    def evaluate(self, line: torch.Tensor) -> torch.Tensor:
        hidden = self.init_hidden()
        with torch.no_grad():
            for i in range(line.size()[0]):
                output, hidden = self(line[i], hidden)
        return output


class Lstm(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(Lstm, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size + output_size, output_size)
        self.fc2 = nn.Linear(input_size + output_size, output_size)
        self.fc3 = nn.Linear(input_size + output_size, output_size)
        self.fc4 = nn.Linear(input_size + output_size, output_size)


    def forward(self, i: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ih = torch.cat((i, h), 1)
        o1 = torch.sigmoid(self.fc1(ih))
        o2 = torch.sigmoid(self.fc2(ih))
        o3 = torch.tanh(self.fc3(ih))
        o4 = torch.sigmoid(self.fc4(ih))
        cxo1 = c * o1
        o2xo3 = o2 * o3
        new_c = cxo1 + o2xo3
        new_h = torch.tanh(c) * o4
        return new_h, new_c


    def init_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, self.output_size), torch.zeros(1, self.output_size)


    def train(self, category: torch.Tensor, line: torch.Tensor, optimizer: optim.Optimizer, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, float]:
        optimizer.zero_grad()
        output, cell = self.init_hidden()

        for i in range(line.size()[0]):
            output, cell = self(line[i], output, cell)

        loss = criterion(output, category)
        loss.backward()
        optimizer.step()
        return output, loss.item()


    def evaluate(self, line: torch.Tensor) -> torch.Tensor:
        output, cell = self.init_hidden()
        with torch.no_grad():
            for i in range(line.size()[0]):
                output, cell = self(line[i], output, cell)
        return output


class Gru(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(Gru, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size + output_size, output_size)
        self.fc2 = nn.Linear(input_size + output_size, output_size)
        self.fc3 = nn.Linear(input_size + output_size, output_size)


    def forward(self, i: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ih = torch.cat((i, h), 1)
        o1 = torch.sigmoid(self.fc1(ih))
        hxo1 = h * o1
        o2 = torch.sigmoid(self.fc2(ih))
        o3 = torch.tanh(self.fc3(torch.cat((hxo1, i), 1)))
        h = (1 - o2) * h + o2 * o3
        return h


    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(1, self.output_size)


    def train(self, category: torch.Tensor, line: torch.Tensor, optimizer: optim.Optimizer, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, float]:
        optimizer.zero_grad()
        output = self.init_hidden()

        for i in range(line.size()[0]):
            output = self(line[i], output)

        loss = criterion(output, category)
        loss.backward()
        optimizer.step()
        return output, loss.item()


    def evaluate(self, line: torch.Tensor) -> torch.Tensor:
        output = self.init_hidden()
        with torch.no_grad():
            for i in range(line.size()[0]):
                output = self(line[i], output)
        return output


def main() -> None:
    category_lines = {}

    for filename in find_files('.data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        lines = read_lines(filename)
        category_lines[category] = lines

    all_categories = list(category_lines.keys())
    n_categories = len(all_categories)

    n_epochs = 100_000
    print_every = n_epochs // 10
    plot_every = n_epochs // 20
    learning_rate = 0.0005
    criterion = nn.CrossEntropyLoss()

    # model = Rnn(n_letters, 128, n_categories)
    # model = Lstm(n_letters, n_letters)
    model = Gru(n_letters, n_letters)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    current_loss = 0.0
    all_losses = []

    start = time.time()
    for e in range(1, n_epochs + 1):
        category, line, category_tensor, line_tensor = random_training_example(all_categories, category_lines)
        output, loss = model.train(category_tensor, line_tensor, optimizer, criterion)
        current_loss += loss

        if e % print_every == 0:
            guess, _ = category_from_output(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (e, e / n_epochs * 100, time_since(start), loss, line, guess, correct))

        if e % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    plot_losses(all_losses)

    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10_000

    for _ in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_example(all_categories, category_lines)
        output = model.evaluate(line_tensor)
        _, guess_i = category_from_output(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    plot_confusion(all_categories, confusion)


main()
