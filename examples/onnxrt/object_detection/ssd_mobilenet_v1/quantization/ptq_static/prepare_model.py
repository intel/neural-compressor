import argparse
import os
import subprocess
import sys
import tarfile
from urllib import request


MODEL_URL = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz"
MAX_TIMES_RETRY_DOWNLOAD = 5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",
                        type=str,
                        required=False,
                        default="ssd_mobilenet_v1_coco_2018_01_28.tar.gz")
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


def extrafile(filename, target_folder="."):
    tf = tarfile.open(filename)
    tf.extractall(target_folder)


def is_tar_gz_file(filename):
    with open(filename, 'rb') as f:
        magic_number = f.read(2)
    return magic_number == b'\x1f\x8b'


def download_model(url, model_name, retry_times=5):
    if os.path.isdir(model_name):
        return model_name
    elif os.path.exists(model_name) and is_tar_gz_file(model_name):
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


def export_model(input_model, output_model):
    # Please refer to [Converting SSDMobilenet To ONNX Tutorial](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb) for detailed model converted.
    print("\nexport model...")
    subprocess.run(
        ["pip", "install", "tf2onnx", "tensorflow"],
        stdout=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "tf2onnx.convert",
            "--graphdef",
            "ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb",
            "--opset",
            "14",
            "--output",
            output_model,
            "--inputs",
            "image_tensor:0",
            "--outputs",
            "num_detections:0,detection_boxes:0,detection_scores:0,detection_classes:0",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )


def prepare_model(input_model, output_model):
    is_download_successful = download_model(MODEL_URL, input_model, MAX_TIMES_RETRY_DOWNLOAD)
    if is_download_successful:
        export_model(input_model, output_model)


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model)