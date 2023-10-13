import os
import tqdm
import onnx
import torch
import logging
import argparse
import onnxruntime as ort
from timm.utils import accuracy
from torchvision import datasets, transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

def build_eval_transform(input_size=224, imagenet_default_mean_and_std=False, crop_pct=None):
    resize_im = input_size > 32
    imagenet_default_mean_and_std = imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    t = []
    if resize_im:
        if crop_pct is None:
            if input_size < 384:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
        size = int(input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_val_dataset(data_path):
    transform = build_eval_transform()
    root = os.path.join(data_path, 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def evaluate_func(data_loader, model):
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    top1, top5 = 0, 0

    for idx, batch in tqdm.tqdm(enumerate(data_loader), desc='eval'):
        images = batch[0].cpu().detach().numpy()
        target = batch[-1]
        output = session.run(None, {'image': images})[0]
        acc1, acc5 = accuracy(torch.from_numpy(output), target, topk=(1, 5))
        top1 += acc1.cpu().detach().numpy()
        top5 += acc5.cpu().detach().numpy()
        
    top1 = top1 / len(data_loader)
    top5 = top5 / len(data_loader)
    print('* Acc@1 {:.3f} Acc@5 {:.3f}'.format(top1, top5))
    return top1

if __name__ == '__main__':
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    parser = argparse.ArgumentParser(
        description="BEiT quantization examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help="Pre-trained model on onnx file"
    )
    parser.add_argument(
        '--dataset_location',
        type=str,
        help="Imagenet data path"
    )
    parser.add_argument(
        '--benchmark',
        action='store_true', \
        default=False,
        help="whether bechmark the model"
    )
    parser.add_argument(
        '--tune',
        action='store_true', \
        default=False,
        help="whether quantize the model"
    )
    parser.add_argument(
        '--output_model',
        type=str,
        help="output model path"
    )
    parser.add_argument(
        '--mode',
        type=str,
        help="benchmark mode of performance or accuracy"
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
    )
    args = parser.parse_args()

    val_dataset = build_val_dataset(args.dataset_location)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        drop_last=False
    )
    
    def eval(model):
        return evaluate_func(val_data_loader, model)
    
    model = onnx.load(args.model_path)

    if args.tune:
        from neural_compressor import PostTrainingQuantConfig, quantization
        from neural_compressor.utils.constant import FP32

        config = PostTrainingQuantConfig(approach="static",
                                         op_type_dict={'^((?!(MatMul|Conv)).)*$': FP32},
                                        quant_level=1,
                                        )
        q_model = quantization.fit(model, config, calib_dataloader=val_data_loader, eval_func=eval)
        q_model.save(args.output_model)

    if args.benchmark:
        if args.mode == 'performance':
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig(warmup=10, iteration=1000, cores_per_instance=4, num_of_instance=1)
            fit(model, conf, b_dataloader=val_data_loader)
        elif args.mode == 'accuracy':
            acc_result = evaluate_func(val_data_loader, model)
            print("Batch size = %d" % val_data_loader.batch_size)
            print("Accuracy: %.5f" % acc_result)
        

