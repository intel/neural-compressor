import os
import numpy as np

import torch.optim as optim

import alexnet

def main():
    train_loader, test_loader = alexnet.data_loader()

    model = alexnet.Net()
    optimizer = optim.Adam(model.parameters(), lr = 0.002)

    epochs = 1
    alexnet.train_mod(model, train_loader, test_loader, optimizer, epochs)
    alexnet.test_mod(model, test_loader)

    alexnet.save_mod(model, "alexnet_mnist_fp32_mod.pth")

if __name__ == "__main__":
    main()
    