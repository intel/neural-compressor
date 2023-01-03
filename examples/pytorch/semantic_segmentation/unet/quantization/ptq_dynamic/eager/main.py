import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--data', type=str, help='Load data from a file')
    parser.add_argument('--tuned_checkpoint', default='./saved_results', type=str, metavar='PATH',
                                    help='path to checkpoint tuned by Neural Compressor (default: ./)')
    parser.add_argument(
        "--is_relative",
        type=bool,
        default="True",
        help="Metric tolerance mode, True for relative, otherwise for absolute.",
    )
    parser.add_argument(
        "--perf_tol",
        type=float,
        default=0.01,
        help="Performance tolerance when optimizing the model.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Only test performance for model.",
    )
    parser.add_argument(
        "--accuracy_only",
        action="store_true",
        help="Only test accuracy for model.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="benchmark for int8 model",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Whether or not to apply quantization.",
    )
    parser.add_argument(
        "--quantization_approach",
        type=str,
        default="static",
        help="Quantization approach. Supported approach are static, "
                  "dynamic and auto.",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    state_dict = torch.load(args.load, map_location=torch.device('cpu'))
    if 'mask_values' in state_dict:
        state_dict.pop('mask_values')
    model.load_state_dict(state_dict)
    
    #model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
    model = model.to(memory_format=torch.channels_last)

    dir_img = Path(args.data + '/imgs')
    dir_mask = Path(args.data + '/masks')

    img_scale = 0.5
    val_percent = 0.1
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    # 4. define INC evaluate function
    def eval_func(model):       
        val_score = evaluate(model, val_loader, device, args.amp)
        return float(val_score)
    
    def calib_func(model):
        for index, batch in enumerate(val_loader):
            image, mask_true = batch['image'], batch['mask']
            model(image)
            if index == 50:
                break
    # 5. Tune
    if args.tune:
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization('conf.yaml')
        quantizer.eval_func = eval_func
        quantizer.calib_func = calib_func
        quantizer.calib_dataloader = val_loader
        quantizer.model = common.Model(model)
        q_model = quantizer.fit()
        q_model.save(args.tuned_checkpoint)

    # 6. Benchmark and accuracy
    if args.int8:
        from neural_compressor.utils.pytorch import load
        model = load(os.path.abspath(os.path.expanduser(args.tuned_checkpoint)), model)
    
    if args.benchmark or args.accuracy_only:
        eval_func(model)

        
