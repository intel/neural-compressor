import argparse
import os
import sys
import torch
from urllib import request
from timm.models import create_model
import beit_modeling_finetune

MODEL_URLS = {"beit_base_patch16_224": "https://github.com/addf400/files/releases/download/v1.0/beit_base_patch16_224_pt22k_ft22kto1k.pth",}
MODEL_FILES = {"beit_base_patch16_224": "beit_base_patch16_224_pt22k_ft22kto1k.pth"}
MAX_TIMES_RETRY_DOWNLOAD = 5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default="beit_base_patch16_224")
    parser.add_argument("--output_model", type=str, required=True)
    return parser.parse_args()


def progressbar(cur, total=100):
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write("\r[%-100s] %s" % ('#' * int(cur), percent))
    sys.stdout.flush()


def schedule(blocknum, blocksize, totalsize):
    if totalsize == 0:
        percent = 0
    else:
        percent = min(1.0, blocknum * blocksize / totalsize) * 100
    progressbar(percent)


def download_model(input_model, retry_times=5):
    model_url = MODEL_URLS[input_model]
    model_file = MODEL_FILES[input_model]
    if os.path.isfile(model_file):
        print(f"{model_file} exists, skip download")
        return True

    print("download model...")
    retries = 0
    while retries < retry_times:
        try:
            request.urlretrieve(model_url, model_file, schedule)
            break
        except KeyboardInterrupt:
            return False
        except:
            retries += 1
            print(f"Download failed{', Retry downloading...' if retries < retry_times else '!'}")
    return retries < retry_times


def export_model(input_model, output_model):
    print("\nexport model...")

    model = create_model(
        input_model,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=True,
        use_abs_pos_emb=False,
        init_values=0.1,
        )

    checkpoint = torch.load(MODEL_FILES[input_model], map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print("Resume checkpoint %s" % MODEL_FILES[input_model])

    model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch.onnx.export(model,
                      x,
                      output_model,
                      export_params=True,
                      opset_version=13,
                      do_constant_folding=True,
                      input_names = ["image"],
                      output_names = ["output"],
                      dynamic_axes={"image" : {0 : "batch_size"},
                                    "output" : {0 : "batch_size"}}
                    )
    assert os.path.exists(output_model), f"Export failed! {output_model} doesn't exist!"


def prepare_model(input_model, output_model):
    is_download_successful = download_model(args.input_model, MAX_TIMES_RETRY_DOWNLOAD)
    if is_download_successful:
        export_model(input_model, output_model)


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model)
