# mpi_hello.py
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD


size = comm.Get_size()
rank = comm.Get_rank()
print(f"I am rank {comm.rank}, the MPI world has {comm.size} peers.")

def worker(rank, size):
    print(f"I am rank {rank}, the MPI world has {size} peers.")
    import os
    import time
    import tqdm
    import torch
    import torchvision
    from torchvision import transforms    
    from torch.utils.data import DataLoader
    model = torchvision.models.resnet18()
    
    print(f"Torch parallel info:")
    print(torch.__config__.parallel_info())
    omp_var = os.environ.get('OMP_NUM_THREADS', None)
    print(f"OMP_NUM_THREADS: {omp_var}")
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),  (0.2675, 0.2565, 0.2761))
    ])
    
    dataset = torchvision.datasets.CIFAR100(root='./cifar100', train=True, download=True, transform=transform_test)
    data_loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    start = time.time()
    for data, label in tqdm.tqdm(data_loader, desc = "Rank @@ "+ str(rank) + " @@" ):
        _ = model(data)
    end = time.time()
    print(f"I am rank {rank}/{size}, token {end - start} s.")

# send and receive message(numpy.array, integrate)
worker(rank, size)
