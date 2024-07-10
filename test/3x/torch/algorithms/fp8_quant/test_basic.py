import os
import sys
import time

import habana_frameworks.torch.core as htcore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        out = x.view(-1, 28 * 28)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        return out


def test_hpu():
    model = Net()
    model_link = "https://vault.habana.ai/artifactory/misc/inference/mnist/mnist-epoch_20.pth"
    model_path = "/tmp/.neural_compressor/mnist-epoch_20.pth"
    os.system("mkdir -p /tmp/.neural_compressor && wget {} -O {} ".format(model_link, model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    model = model.eval()

    model = model.to("hpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    data_path = "./data"
    test_kwargs = {"batch_size": 32}
    dataset1 = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset1, **test_kwargs)

    correct = 0
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to("hpu")
        output = model(data)
        htcore.mark_step()
        correct += output.max(1)[1].eq(label).sum()

    accuracy = 100.0 * correct / (len(test_loader) * 32)
    assert accuracy > 90
