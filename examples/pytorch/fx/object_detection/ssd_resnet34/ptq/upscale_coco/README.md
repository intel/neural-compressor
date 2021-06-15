# Upscaled COCO Dataset
Upscale COCO images for MLPerf Inference.

## Description
The python script performs the following operations on images: read input images, upscale images, save images to file; and on annotations: read json file, modify annotations, write annotations to file. The input files consist of the parent directory, image directory and path to annotations file. The upscaled images and modified annotations file are written to the specified output parent directory using the same directory structure as the inputs.

## Parameters
- inputs: parent directory for input coco images and annotations.
- outputs: parent directory for upscaled coco images and annotations.
- images: path to coco images.
- annotations: path to annotations json file.
- size: upscaled image sizes.
- format: upscaled image format.

## How to run
1. Upscale data set to 700 x 700 images:
   ```
   python upscale_coco.py --inputs /coco --outputs ./coco700 --images val2017 --annotations annotations/instances_val2017.json --size 700 700
   ```

2. Upscale data set to 700 x 700 png images:
   ```
   python upscale_coco.py --inputs /coco --outputs ./coco700 --images val2017 --annotations annotations/instances_val2017.json --size 700 700 --format png
   ```

## Notes
Images stored as jpeg result in up to 0.5 lower mAP score because of lossy compression. Need to store images in png to obtain same results as for MLPerf training.
