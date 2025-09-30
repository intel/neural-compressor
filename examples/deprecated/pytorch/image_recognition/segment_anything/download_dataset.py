import torchvision

print("Downloading VOC dataset")
torchvision.datasets.VOCDetection(root='./voc_dataset', year='2012', image_set ='trainval', download=True)



