import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from neural_compressor.utils.pytorch import load
import fashion_mnist


class Net(nn.Module):
    def __init__(self, num_classes = 10, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)          
        )        
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),            
            nn.LogSoftmax(dim=1)            
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)        
        x = torch.flatten(x, 1)
        x = self.classifier(x)        
        return x
        
def save_mod(model, model_file):
    print('Save to {}'.format(model_file))
    torch.save(model.state_dict(), model_file)

def load_mod(model_file):
    model = Net()    
    model.load_state_dict(torch.load(model_file))
    print('Load from {}'.format(model_file))
    return model

def load_int8_mod(model_folder):
    model = Net()
    int8_model = load(model_folder, model)    
    print('Load from {}'.format(model_folder))
    return int8_model
    
def data_loader(batch_size=200):
    train_loader, test_loader = fashion_mnist.data_loader(batch_size)
    return train_loader, test_loader

def do_test_mod(model, test_loader):    
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
      model.eval()
      for images, labels in test_loader:
        log_ps = model(images)
        test_loss += F.nll_loss(log_ps, labels)

        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
    accuracy = (accuracy/len(test_loader)).numpy().item()
    test_loss = (test_loss/len(test_loader)).numpy().item()
    
    return test_loss, accuracy

def test_mod(model, test_loader):
    print("Testing ...")
    test_loss, accuracy = do_test_mod(model, test_loader)
    print("Test loss: {:.3f}..".format(test_loss),
          "Test Accuracy: {:.3f}".format(accuracy))


def train_mod(model, train_loader, test_loader, optimizer, epochs=3):
    print("Training ...")
    model.train()
    running_loss = 0
    train_len = len(train_loader)
    
    for epoch in range(1, epochs + 1):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model.forward(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            print("Epoch {}/{} Iteration {}/{} Loss {:.6f}".format(epoch, epochs, i, train_len, \
                running_loss/(i+1)), end='\r')
            
            
        test_loss, accuracy = do_test_mod(model, test_loader)
        
        print('\nTrain Epoch: {} Epoch {} Samples \tLoss: {:.6f} Test Loss: {:.6f} Accuracy: {:.6f}'.format(
            epoch, len(train_loader.sampler),
            running_loss/len(train_loader), test_loss, accuracy))
    print("Done")

def main():
    train_loader, test_loader = data_loader()

    model = Net()
    print(model)

    optimizer = optim.Adam(model.parameters(), lr = 0.002)

    epochs = 1
    train_mod(model, train_loader, test_loader, optimizer, epochs)
    test_mod(model, test_loader)

    save_mod(model, "alexnet_mnist_fp32_mod.th")

if __name__ == "__main__":
    main()
