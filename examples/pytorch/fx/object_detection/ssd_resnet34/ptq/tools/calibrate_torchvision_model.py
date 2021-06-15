import argparse
import os
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from torchvision.models.quantization import *


class CalibrationDataset(Dataset):
    def __init__(self, root, files, transform):
        with open(files, 'r') as f:
            self.files = [os.path.join(root, fn.strip()) for fn in f.readlines()]
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.files)


def quantize_model(model, dataloader, backend='fbgemm'):
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.cpu()
    model.eval()
    model.fuse_model()

    # Make sure that weight qconfig matches that of the serialized models
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)
    print('calibrating...')
    for x in tqdm(dataloader):
        model(x)
    print('calibration DONE!')
    torch.quantization.convert(model, inplace=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--image-dir', type=str, default='imagenet/val')
    parser.add_argument('--image-list', type=str, default='../../calibration/ImageNet/cal_image_list_option_1.txt')
    args = parser.parse_args()
    print(args)

    transform = transforms.Compose([                                                   
        transforms.Resize(256),                                                        
        transforms.CenterCrop(224),                                                    
        transforms.ToTensor(),                                                         
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   
    ])                                                                                 

    dataset = CalibrationDataset(root=args.image_dir, files=args.image_list, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)

    model = eval(args.model)(pretrained=True, progress=True, quantize=False)
    quantize_model(model, dataloader)
    print(model)

    inp = torch.rand(1, 3, 224, 224)
    script_module = torch.jit.trace(model, inp)
    save_path = f'{args.model}.pt'
    torch.jit.save(script_module, save_path)
    print(f'saved: {save_path}')


if __name__=='__main__':
    main()

