from PIL import Image
import torch
import glob
from tqdm import tqdm
import argparse
from torchmetrics import PSNR #PeakSignalNoiseRatio as PSNR
from torchmetrics import SSIM #StructuralSimilarityIndexMeasure as SSIM
import numpy as np
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser('evaluate Super Resolution using SSIM and PSNR', add_help=False)
parser.add_argument('--num_images', default=10, type=int, help='number of images to use')
parser.add_argument('--real_images', type = str, default = '/datasets/imagenet/val_cropped_labeled', help = 'path to real images')
parser.add_argument('--gen_images', type = str, help = 'path to real images')

args = parser.parse_args()
num_images = args.num_images
gen_image_path = args.gen_images + '/*'
real_image_path = args.real_images + '/*'

# define transform PIL to tensor, used later in loop because 
# metrics need to receive tensor 
transform = transforms.Compose([transforms.PILToTensor()])

# import metrics 
psnr = PSNR().to(device)
ssim = SSIM().to(device)

# list of metric for each image 
psnr_distances = []
ssim_distances = []

# iterate over all of the files in the folders and calculate metric
files_real = sorted(glob.iglob(real_image_path))
files_gen = sorted(glob.iglob(gen_image_path))


with torch.no_grad():
    for i in tqdm(np.arange(num_images)):
        real_image = Image.open(files_real[i])
        real_image = transform(real_image).to(device).to(torch.float)

        gen_image = Image.open(files_gen[i])
        gen_image = transform(gen_image).to(device).to(torch.float)

        psnr_res = psnr(real_image, gen_image)
        psnr_res = psnr_res.item()
        psnr_distances.append(psnr_res)
        
        ssim_res = ssim(torch.unsqueeze(real_image, dim=0),torch.unsqueeze(gen_image, dim=0))
        ssim_res = ssim_res.item()
        ssim_distances.append(ssim_res)
    
        # to avoid out of memory
        ssim.reset()
        psnr.reset()

    # turn list into a numpy array to calcultate average
    try:
        psnr_distance = np.array(psnr_distances)
        ssim_distance = np.array(ssim_distances)
    except:
        print(f"error: no files in requested path")
        quit()

#calculate mean and print
print(f"mean psnr is {np.mean(psnr_distance)}")
print(f"mean ssim is {np.mean(ssim_distance)}")
