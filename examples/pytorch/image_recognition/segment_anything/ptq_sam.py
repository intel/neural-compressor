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
from inc_dataset_loader import INC_SAMVOC2012Dataset
from neural_compressor.data import DataLoader
from neural_compressor.quantization import fit
from functools import partial


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)
    
def apply_coords(coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    target_length = 1024
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(original_size[0], original_size[1], target_length)
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords

def apply_boxes(boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
    """
    Expects a numpy array shape Bx4. Requires the original image size
    in (H, W) format.
    """
    boxes = apply_coords(boxes.reshape(-1, 2, 2), original_size)
    return boxes.reshape(-1, 4)

# Adapted from SAM's implementation
# Ref: https://github.com/facebookresearch/segment-anything/blob/c1910835a32a05cbb79bdacbec8f25914a7e3a20/segment_anything/modeling/sam.py#L133
def postprocess_masks(
    masks: torch.Tensor,
    input_size: Tuple[int, ...],
    original_size: Tuple[int, ...],
) -> List[torch.Tensor]:
    """
    Remove padding and upscale masks to the original image size.

    Arguments:
        masks (torch.Tensor): Batched masks from the mask_decoder,
        in BxCxHxW format.
        input_size (tuple(int, int)): The size of the image input to the
        model, in (H, W) format. Used to remove padding.
        original_size (tuple(int, int)): The original size of the image
        before resizing for input to the model, in (H, W) format.

    Returns:
        (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.
    """
    image_encoder_img_size = 1024

    masks = F.interpolate(
        masks,
        (image_encoder_img_size, image_encoder_img_size),
        mode="bilinear",
        align_corners=False,
    )
    
    unpadded_mask = masks[..., : input_size[0],  : input_size[1]]
    #unpadded_mask = unpadded_mask[None, :, : , :] # Add batch information
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
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
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
        #x: Dict[str, Any],
        #multimask_output: bool,
    ):
        
        #Encode the images
        if len(image.shape) == 3:
            image = image[None, ...] # Append batch information for image_encoder

        image_embeddings = self.image_encoder(image)

        
        input = np.zeros(4)
        input[0] = prompt[0] #x['prompt'][0].item()
        input[1] = prompt[1] #x['prompt'][1].item()
        input[2] = prompt[2] #x['prompt'][2].item()
        input[3] = prompt[3] #x['prompt'][3].item()
        #original_size_tuple = (x['original_size'][0][0].item(), x['original_size'][1][0].item()) # H, W

        #import pdb; pdb.set_trace()
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
            #multimask_output=multimask_output,
            multimask_output=False,
        )

        #Post process
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=input_size, #x['input_size'],
            original_size=original_size_tuple,
        )

        masks = masks > self.mask_threshold
        # outputs.append(
        #     {
        #         "masks": masks,
        #         "iou_predictions": iou_predictions,
        #         "low_res_logits": low_res_masks,
        #     }
        # )
        #return outputs

        #import pdb; pdb.set_trace()
        #return masks[0].int() #Output the pred for dataloader to compare
        return masks[0].int() # Output pred for dataloader to comapre

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
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
        """Normalize pixel values and pad to a square input."""
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


# Start PTQ
if __name__ == '__main__':
    model = Sam_INC() 
    model.load_state_dict(torch.load('./sam_vit_b_01ec64.pth'))
    train_inc_dataset = INC_SAMVOC2012Dataset('./voc_dataset/VOCdevkit/VOC2012/', 'train')
    eval_inc_dataset = INC_SAMVOC2012Dataset('./voc_dataset/VOCdevkit/VOC2012/', 'val')
    calib_dataloader = DataLoader(framework="pytorch", dataset=train_inc_dataset)
    eval_dataloader = DataLoader(framework="pytorch", dataset=eval_inc_dataset)
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
    config = PostTrainingQuantConfig(op_type_dict=op_type_dict)
    q_model = fit(model, config, calib_dataloader=calib_dataloader, eval_dataloader=eval_dataloader, eval_func=eval_func)
    import pdb; pdb.set_trace()