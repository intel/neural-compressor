import argparse
import os
import subprocess
import sys
from urllib import request

MODEL_URLS = {"structure_detr": "https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_structure_detr_r18.pth",
              "detection_detr": "https://huggingface.co/bsmock/tatr-pubtables1m-v1.0/resolve/main/pubtables1m_detection_detr_r18.pth"}
MAX_TIMES_RETRY_DOWNLOAD = 5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",
                        type=str,
                        required=False,
                        choices=["structure_detr", "detection_detr"],
                        default="structure_detr")
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--dataset_location", type=str, required=True)
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


def download_model(url, retry_times=5):
    model_name = url.split("/")[-1]
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


def export_model(input_model, output_model, dataset_location):
    print("\nexport model...")

    if not os.path.exists("./table-transformer"):
        subprocess.run("bash prepare.sh", shell=True)
    
    model_load_path = os.path.abspath(MODEL_URLS[input_model].split("/")[-1])
    output_model = os.path.join(os.path.dirname(model_load_path), output_model)
    if input_model == "detection_detr":
        data_root_dir = os.path.join(dataset_location, "PubTables-1M-Detection")
        data_type = "detection"
        config_file = "detection_config.json"
    elif input_model == "structure_detr":
        data_root_dir = os.path.join(dataset_location, "PubTables-1M-Structure")
        data_type = "structure"
        config_file = "structure_config.json"
    table_words_dir = os.path.join(data_root_dir, "words")

    os.chdir("table-transformer/src")

    command = f"python main.py \
                --model_load_path {model_load_path} \
                --output_model {output_model} \
                --data_root_dir {data_root_dir} \
                --table_words_dir {table_words_dir} \
                --mode export \
                --data_type {data_type} \
                --device cpu \
                --config_file {config_file}"

    subprocess.run(command, shell=True)
    assert os.path.exists(output_model), f"Export failed! {output_model} doesn't exist!"


def prepare_model(input_model, output_model, dataset_location):
    is_download_successful = download_model(MODEL_URLS[args.input_model], MAX_TIMES_RETRY_DOWNLOAD)
    if is_download_successful:
        export_model(input_model, output_model, dataset_location)


if __name__ == "__main__":
    args = parse_arguments()
    prepare_model(args.input_model, args.output_model, args.dataset_location)