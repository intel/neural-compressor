import argparse
import os
import sys
from urllib import request

import onnx
from onnx import version_converter

MODEL_URL = "https://github.com/onnx/models/raw/main/archive/vision/body_analysis/ultraface/models/version-RFB-320.onnx"
MAX_TIMES_RETRY_DOWNLOAD = 5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default='version-RFB-640.onnx')
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


def download_model(url, model_name, retry_times=5):
    if os.path.isfile(model_name):
        print(f"{model_name} exists, skip download")
        return True

    print("download model...")
    retries = 0
    while retries < retry_times:
        try:
            request.urlretrieve(url, model_name, schedule)
            break
        except KeyboardInterrupt:
            return False
        except:
            retries += 1
            print(f"Download failed{', Retry downloading...' if retries < retry_times else '!'}")
    return retries < retry_times


def export_model(input_model, output_model):
    # Convert opset version to 14 for more quantization capability.
    print("\nexport model...")
    model = onnx.load(input_model)
    model = version_converter.convert_version(model, 14)
    onnx.save_model(model, output_model)
    assert os.path.exists(output_model), f"Export failed! {output_model} doesn't exist!"


def prepare_model(input_model, output_model):
    # Download model from [ONNX Model Zoo](https://github.com/onnx/models).
    is_download_successful = download_model(MODEL_URL, input_model, MAX_TIMES_RETRY_DOWNLOAD)
    if is_download_successful:
        export_model(input_model, output_model)


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model)