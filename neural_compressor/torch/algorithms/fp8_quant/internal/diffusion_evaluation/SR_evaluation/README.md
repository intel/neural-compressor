How to calculate PSNR and SSIM for Super Resolution
We will use the Imagenet validation dataset.

The evaluation is done by the following steps:
1) We take the Imagenet validation set which has 50,000 images (We can also take a subset) 
2) Crop these Images to be 256*256 (center cropped), and save these images as the "ground truth" dataset. The name of 
the saved image is its label.
3) Downsample the images to be 64*64 (using bicubic interpolation) and then restore them using Super Resolution. 
4) Calculate  PSNR and SSIM between each ground truth image and restored image, and print the mean.

Steps 1,2 and 4 are included here, while step 3 (downsampling and restoring) should be done separately, using the 
desired Super Resolution method. Keep in mind that this script assumes that the images are stored in a specific format, 
(detailed later). Later, the restored images path should be given as an input to step 4.

You can skip step 1+2 and use the images at /datasets/imagenet/val_cropped_labeled
You can also run a python script which does the following to the imagenet validation dataset:
 - Crops images to 256*256 (this can also be changed using the argument --resize, 256*256 is the default)
 - Saves the images with the convention <path>/<label>_<ID>.png
 - a text file mapping imagenet class index to label is needed. It is given here as imagenet1000_clsidx_to_labels.txt, but 
 can be given as an argument with --class_to_labels

to do this, run the following: 

python create_SR_dataset.py --images <imagenet validation path> --out_dir <path to save ground truth images>

Now, create the generated images so they match the files created above (step 3)

IMPORTANT!! - the script that does the actual evaluation (explained below) expects to get an image where the prompt in the same format 
<generated images path>/<label>_<ID>.png. This means that the script expects the original and restored images to have the same name.

Find an example in /workdisk/ilamprecht/diffusion/stablediffusionv2/scripts/superres_gen_imgs.py

Now, run the evaluation script, which calculates PSNR and SSIM and prints the mean (step 4)

To do this, run:

python super_res_eval.py --num_images <desired number of images up to 50000> --real_images <real images path> --gen_images <generated images path>
