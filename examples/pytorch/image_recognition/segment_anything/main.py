from segment_anything import SamPredictor, sam_model_registry
import torchvision
import torch
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from statistics import mean
import torch
from torch import nn
from torch.nn.functional import threshold, normalize
import torch.nn.functional as F
from typing import List, Tuple
from copy import deepcopy
import torchmetrics
from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, TwoWayTransformer
from typing import Dict, Any
from neural_compressor import quantization, PostTrainingQuantConfig
from neural_compressor.config import TuningCriterion, AccuracyCriterion
from inc_dataset_loader import INC_SAMVOC2012Dataset
from neural_compressor.data import DataLoader
from neural_compressor.quantization import fit
from functools import partial
import argparse

# Preprocessing codes are adapted from original SAM's implementation
# Ref: https://github.com/facebookresearch/segment-anything/blob/c1910835a32a05cbb79bdacbec8f25914a7e3a20/segment_anything/modeling/sam.py

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)


def apply_coords(coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
    target_length = 1024
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(original_size[0], original_size[1], target_length)
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords


def apply_boxes(boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
    boxes = apply_coords(boxes.reshape(-1, 2, 2), original_size)
    return boxes.reshape(-1, 4)


def postprocess_masks(
    masks: torch.Tensor,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
) -> List[torch.Tensor]:
    image_encoder_img_size = 1024

    masks = F.interpolate(
        masks,
        (image_encoder_img_size, image_encoder_img_size),
        mode="bilinear",
        align_corners=False,
    )
    
    unpadded_mask = masks[..., : input_size[0],  : input_size[1]]
    mask = F.interpolate(unpadded_mask, original_size, mode="bilinear", align_corners=False)
    mask = mask[0] #Remove the unnecessary batch dimension

    return mask


class Sam_INC(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()

        # Moved from _build_sam() 
        # Specs for build_sam_vit_b
        encoder_embed_dim=768
        encoder_depth=12
        encoder_num_heads=12
        encoder_global_attn_indexes=[2, 5, 8, 11]

        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )

        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        image,
        prompt,
        original_size,
        input_size,
        ground_truth_mask,
    ):
        
        #Encode the images
        if len(image.shape) == 3:
            image = image[None, ...] # Append batch information for image_encoder

        image_embeddings = self.image_encoder(image)

        
        input = np.zeros(4)
        input[0] = prompt[0]
        input[1] = prompt[1]
        input[2] = prompt[2]
        input[3] = prompt[3]

        original_size_tuple = (original_size[0].item(), original_size[1].item()) # H, W
        transformed_boxes = apply_boxes(input, original_size_tuple)
        transformed_boxes = torch.as_tensor(transformed_boxes, dtype=torch.float)
        transformed_boxes = transformed_boxes[None, :]

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, # Ignore point
            boxes=transformed_boxes, #Take only 1 box as input
            masks=None, # Ignore mask
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings[0],
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        #Post process
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=input_size, #x['input_size'],
            original_size=original_size_tuple,
        )

        masks = masks > self.mask_threshold
        return masks[0].int() # Output pred for dataloader to comapre

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


def eval_func(model):
    device = 'cpu'
    metric = torchmetrics.Dice(ignore_index=0).to(device) #Ignore background
    list_of_metrics = []
    
    for i, (input, gt) in enumerate(eval_dataloader):
        preds = model(input['image'],
                      input['prompt'],
                      input["original_size"],
                      input["input_size"],
                      input["ground_truth_mask"])
        labels = gt
        result = metric(preds.reshape(-1), labels.reshape(-1).int())
        list_of_metrics.append(result)

    return np.array(list_of_metrics).mean()


def validate(eval_dataloader, model, args):
    model.eval()
    device = 'cpu'
    metric = torchmetrics.Dice(ignore_index=0).to(device)
    list_of_metrics = []

    with torch.no_grad():
        for i, (input, gt) in enumerate(eval_dataloader):
            preds = model(input['image'],
                        input['prompt'],
                        input["original_size"],
                        input["input_size"],
                        input["ground_truth_mask"])
            labels = gt
            result = metric(preds.reshape(-1), labels.reshape(-1).int())
            list_of_metrics.append(result)

    print("Average Dice Score: " + str(np.array(list_of_metrics).mean()) )
    return
    
    
# Start PTQ
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch VOC Training')
    parser.add_argument("--pretrained_weight_location", default='./sam_vit_b_01ec64.pth', type=str,
                        help='Location of the image encoder pretrained weights')
    parser.add_argument('--voc_dataset_location', default='./voc_dataset/VOCdevkit/VOC2012/', type=str,
                        help='Path of the VOC Dataset')
    parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                        help='path to checkpoint tuned by Neural Compressor (default: ./)')
    parser.add_argument("--tune", action='store_true',
                        help='Apply INT8 quantization or not')   
    parser.add_argument('--int8', action='store_true',
                        help='for benchmarking/validation using quantized model')
    parser.add_argument('--dice', action='store_true',
                        help='For dice score measurements')
    parser.add_argument("--iter", default=0, type=int,
                        help='For dice measurement only.')
    parser.add_argument("--performance", action='store_true',
                        help='For benchmaking')

    args = parser.parse_args()

    # Prepare the model
    model = Sam_INC() 
    model.load_state_dict(torch.load(args.pretrained_weight_location))
    train_inc_dataset = INC_SAMVOC2012Dataset(args.voc_dataset_location, 'train')
    eval_inc_dataset = INC_SAMVOC2012Dataset(args.voc_dataset_location, 'val')
    calib_dataloader = DataLoader(framework="pytorch", dataset=train_inc_dataset)
    eval_dataloader = DataLoader(framework="pytorch", dataset=eval_inc_dataset)

    # quantization
    if args.tune:
        op_type_dict={
            'Embedding':
            {
                'weight': {'dtype': 'fp32'},
                'activation': {'dtype': 'fp32'},
            },
            'ConvTranspose2d':
            {
                'weight': {'dtype': 'fp32'},
                'activation': {'dtype': 'fp32'},
            },
            'Conv2d':
            {
                'weight': {'dtype': 'int8'},
                'activation': {'dtype': 'int8'},
            },
            'Linear':
            {
                'weight': {'dtype': 'int8'},
                'activation': {'dtype': 'int8'},
            },
            'LinearReLU':
            {
                'weight': {'dtype': 'int8'},
                'activation': {'dtype': 'int8'},
            },
            'LayerNorm':
            {
                'weight': {'dtype': 'fp32'},
                'activation': {'dtype': 'fp32'},
            },   
        }

        accuracy_criterion=AccuracyCriterion(tolerable_loss=0.05)
        tuning_criterion=TuningCriterion(timeout=0, max_trials=1)
        config = PostTrainingQuantConfig(op_type_dict=op_type_dict, tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion)
        config.use_bf16=False
        
        q_model = fit(model, config, calib_dataloader=calib_dataloader,  eval_func=eval_func)
        q_model.save(args.tuned_checkpoint)

    # benchmark/evaluation
    if args.int8:
        from neural_compressor.utils.pytorch import load
        new_model = load(os.path.abspath(os.path.expanduser(args.tuned_checkpoint)), model, dataloader=eval_dataloader)
    else:
        new_model = model
            
    if args.performance:
        from neural_compressor.config import BenchmarkConfig
        from neural_compressor import benchmark
        b_conf = BenchmarkConfig(warmup=5,
                                iteration=args.iter,
                                cores_per_instance=52,
                                num_of_instance=1)
        benchmark.fit(new_model, b_conf, b_dataloader=eval_dataloader)
    if args.dice:
        validate(eval_dataloader, new_model, args)
