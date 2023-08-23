import argparse
import os
import sys
import zipfile
from urllib import request

import torch
from transformers import BertForSequenceClassification

# Please refer to [Bert-GLUE_OnnxRuntime_quantization guide]
# (https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/notebooks/bert/Bert-GLUE_OnnxRuntime_quantization.ipynb) 
# for detailed model export.

MODEL_URL = "https://download.pytorch.org/tutorial/MRPC.zip"
MAX_TIMES_RETRY_DOWNLOAD = 5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default="MRPC.zip")
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument('--max_len',
                        type=int,
                        default=128,
                        help='Maximum length of the sentence pairs')
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


def is_zip_file(filename):
    try:
        with open(filename, 'rb') as f:
            magic_number = f.read(4)
            return magic_number == b'PK\x03\x04'  # ZIP file magic number
    except OSError:
        return False


def extrafile(filename, target_folder="."):
    with zipfile.ZipFile(filename, 'r') as zin:
        zin.extractall(target_folder)


def download_model(url, model_name, retry_times=5):
    if os.path.isdir(model_name):
        return model_name
    elif os.path.exists(model_name) and is_zip_file(model_name):
        print("file downloaded")
        extrafile(model_name)
        return True

    print("download model...")
    retries = 0
    while retries < retry_times:
        try:
            request.urlretrieve(url, model_name, schedule)
            extrafile(model_name)
            break
        except KeyboardInterrupt:
            return False
        except:
            retries += 1
            print(f"Download failed{', Retry downloading...' if retries < retry_times else '!'}")
    return retries < retry_times


def export_model(model, output_model, max_len=128):
    with torch.no_grad():
        inputs = {
            'input_ids': torch.ones(1, max_len, dtype=torch.int64),
            'attention_mask': torch.ones(1, max_len, dtype=torch.int64),
            'token_type_ids': torch.ones(1, max_len, dtype=torch.int64)
        }

        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(
            model,  # model being run
            (
                inputs['input_ids'],
                inputs['attention_mask'],
                inputs['token_type_ids'],
            ),  # model input (or a tuple for multiple inputs)
            output_model,  # where to save the model (can be a file or file-like object)
            opset_version=14,  # the ONNX version to export the model
            do_constant_folding=True,  # whether to execute constant folding
            input_names=[
                'input_ids',  # the model's input names
                'input_mask',
                'segment_ids'
            ],
            output_names=['output'],  # the model's output names
            dynamic_axes={
                'input_ids': symbolic_names,  # variable length axes
                'input_mask': symbolic_names,
                'segment_ids': symbolic_names
            })
        print("ONNX Model exported to {0}".format(output_model))


def prepare_model(input_model, output_model, max_len):
    is_download_successful = download_model(MODEL_URL, input_model, MAX_TIMES_RETRY_DOWNLOAD)
    if is_download_successful:
        folder_name = is_download_successful if isinstance(is_download_successful,
                                                           str) else "./MRPC"
        model = BertForSequenceClassification.from_pretrained(folder_name)
        export_model(model, output_model, max_len)


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model, args.max_len)
