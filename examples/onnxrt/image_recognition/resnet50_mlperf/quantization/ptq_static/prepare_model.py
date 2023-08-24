import argparse
import os
import subprocess
import sys
from urllib import request

MODEL_URL = "https://zenodo.org/record/2535873/files/resnet50_v1.pb"
MAX_TIMES_RETRY_DOWNLOAD = 5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, required=False, default="resnet50_v1.pb")
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
    # Use [tf2onnx tool](https://github.com/onnx/tensorflow-onnx) to convert tflite to onnx model.
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
            "--input",
            input_model,
            "--output",
            output_model,
            "--inputs-as-nchw",
            "input_tensor:0",
            "--inputs",
            "input_tensor:0",
            "--outputs",
            "softmax_tensor:0",
            "--opset",
            "14",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert os.path.exists(output_model), f"Export failed! {output_model} doesn't exist!"


def prepare_model(input_model, output_model):
    # Please refer to [MLPerf Inference Benchmarks for Image Classification and Object Detection Tasks]
    # (https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection#mlperf-inference-benchmarks-for-image-classification-and-object-detection-tasks)
    # for model details.
    is_download_successful = download_model(MODEL_URL, input_model, MAX_TIMES_RETRY_DOWNLOAD)
    if is_download_successful:
        export_model(input_model, output_model)


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model)