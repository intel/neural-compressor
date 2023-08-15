import argparse
import os
import sys
import zipfile
from urllib import request

MODEL_URL = {
    "bert": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
    "BERT-Squad": "https://github.com/onnx/models/raw/main/text/machine_comprehension/bert-squad/model/bertsquad-12.onnx",
}
MAX_TIMES_RETRY_DOWNLOAD = 5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default="bert")
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


def is_zip_file(filename):
    try:
        with open(filename, 'rb') as f:
            magic_number = f.read(4)
            return magic_number == b'PK\x03\x04'  # ZIP file magic number
    except OSError:
        return False


def is_onnx_file(filename): 
    with open(filename, 'rb') as f:
        magic_number = f.read(4)
    return magic_number == b'ONNX' # ONNX file magic number


def extrafile(filename, target_folder="."):
    with zipfile.ZipFile(filename, 'r') as zin:
        zin.extractall(target_folder)


def download_model(url, model_name, retry_times=5):
    if os.path.isdir(model_name):
        return model_name
    elif is_zip_file(model_name):
        print("file downloaded")
        extrafile(model_name)
        return True
    elif is_onnx_file(model_name):
        return True

    print("download model...")
    retries = 0
    while retries < retry_times:
        try:
            request.urlretrieve(url, model_name, schedule)
            if is_zip_file(model_name):
                extrafile(model_name)
            break
        except KeyboardInterrupt:
            return False
        except:
            retries += 1
            print(f"Download failed{', Retry downloading...' if retries < retry_times else '!'}")
    return retries < retry_times


def prepare_model(input_model, output_model):
     download_model(MODEL_URL.get(input_model), output_model, MAX_TIMES_RETRY_DOWNLOAD)


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model)
