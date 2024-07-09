import os
import sys
import torch
import time
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F

import habana_frameworks.torch.core as htcore


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1   = nn.Linear(784, 256)
        self.fc2   = nn.Linear(256, 64)
        self.fc3   = nn.Linear(64, 10)
    def forward(self, x):
        out = x.view(-1,28*28)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        return out

model = Net()
checkpoint = torch.load('mnist-epoch_20.pth')
model.load_state_dict(checkpoint)

model = model.eval()

model = model.to("hpu")



model = torch.compile(model,backend="hpu_backend")


transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

data_path = './data'
test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

correct = 0
with torch.no_grad():
    for data, label in test_loader:

        data = data.to("hpu")

        label = label.to("hpu")

        output = model(data)
        correct += output.argmax(1).eq(label).sum().item()

accuracy = correct / len(test_loader.dataset) * 100
print('Inference with torch.compile Completed. Accuracy: {:.2f}%'.format(accuracy))