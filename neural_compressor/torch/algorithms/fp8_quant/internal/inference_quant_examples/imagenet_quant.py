import torch
import torchvision
import numpy as np 
import torchvision.transforms as transforms
import os
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import get_hqt_config, QuantMode
from neural_compressor.torch.quantization import prepare, convert, finalize_calibration, FP8Config

# fp8 additions
import neural_compressor

# data
imgnet_data = '/software/data/pytorch/imagenet/ILSVRC2012/val/'
transform_test = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
                               transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
testset = torchvision.datasets.ImageFolder(imgnet_data, transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


# Define ResNet-18 model
model = torchvision.models.quantization.resnet18(pretrained=True)
# fp8 additions

config_path = os.getenv("QUANT_CONFIG")
config = FP8Config.from_json_file(config_path)
if config.measure:
    model = prepare(model, config)
elif config.quantize:
    model = convert(model, config)

quant_config = get_hqt_config(model).cfg


# evaluate module
device = 'hpu'
model.to(device)
model.eval()



def evaluate():
    accuracy = []
    max_batches = 10 if quant_config['mode'] == QuantMode.MEASURE else 50
    for i,(images,labels) in enumerate(testloader):
        images = images.to(device)
        labels =labels.to(device)
        output = model(images)
        accurate = 0
        total = 0
        _,predicted = torch.max(output.data, 1)
        # total labels
        total+= labels.size(0)
        # Total correct predictions
        accurate += (predicted == labels).sum()
        accuracy_score = 100 * accurate/total
        accuracy.append(accuracy_score)
        if max_batches > 0:
            max_batches -= 1
        else:
            break
        
    accuracy = [x.item() for x in accuracy]
    print(np.mean(np.array(accuracy)))

with torch.no_grad():

    evaluate()

    # fp8 additions
    if quant_config['mode'] == QuantMode.MEASURE:
        finalize_calibration(model)


