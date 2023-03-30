import torch
from torchvision import datasets, transforms

def download_dataset():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
        ])
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                       transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False,
                       transform=transform)
    return train_dataset, test_dataset

def data_loader(batch_size=200):        
    train_dataset, test_dataset = download_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    return train_loader, test_loader

def main():
    train_loader, test_loader = data_loader(batch_size=100)
    print(train_loader.batch_size* len(train_loader))
    print(test_loader.batch_size* len(test_loader))
    
    
if __name__ == "__main__":
    main()
