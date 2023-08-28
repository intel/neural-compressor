import argparse
import os

import requests

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--logs_dir", type=str, default=".")
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--build_id", type=str, default="0")
args = parser.parse_args()
print(args)


def main():
    file_dir = args.logs_dir
    summary_content = ["OS;Platform;Framework;Version;Precision;Model;Mode;Type;BS;Value;Url\n"]
    tuning_info_content = ["OS;Platform;Framework;Version;Model;Strategy;Tune_time\n"]
    url_dict = parse_download_url()
    # get full path of all files
    for root, dirs, files in os.walk(file_dir):
        for name in files:
            file_name = os.path.join(root, name)
            print(file_name)
            if "_summary.log" in name:
                for line in open(file_name, "r"):
                    if "linux" in line:
                        line = line.replace("<url>", parse_summary_log(line, url_dict))
                        summary_content.append(line)
            if "_tuning_info.log" in name:
                for line in open(file_name, "r"):
                    if "linux" in line:
                        line = line.replace("<url>", parse_tuning_log(line, url_dict))
                        tuning_info_content.append(line)
    f = open(args.output_dir + "/summary.log", "a")
    for summary in summary_content:
        f.writelines(str(summary))
    f2 = open(args.output_dir + "/tuning_info.log", "a")
    for tuning_info in tuning_info_content:
        f2.writelines(str(tuning_info))


def parse_tuning_log(line, url_dict):
    """Parsing {Framework}-{Model}-tune.log to get tuning result."""
    result = line.split(";")
    OS, Platform, Framework, Version, Model, Strategy, Tune_time, Tuning_trials, URL, __ = result
    file_name = f"{Framework}-{Model}-tune.log"
    download_url = url_dict.get(f"{Framework}_{Model}")
    download_url = f"{download_url}{file_name}"
    return download_url


def parse_summary_log(line, url_dict):
    """Parse {Framework}-{Model}-tune.log to get benchmarking accuracy result."""
    result = line.split(";")
    OS, Platform, Framework, Version, Precision, Model, Mode, Type, BS, Value, Url = result
    file_name = f"{Framework}-{Model}-tune.log"
    download_url = url_dict.get(f"{Framework}_{Model}")
    download_url = f"{download_url}{file_name}"
    return download_url


def parse_download_url():
    """Get azure artifact information."""
    azure_artifact_api_url = (
        f"https://dev.azure.com/lpot-inc/neural-compressor/_apis/build/builds/{args.build_id}/artifacts?api-version=5.1"
    )
    azure_artifacts_data = dict(requests.get(azure_artifact_api_url).json().items())
    artifact_count = azure_artifacts_data.get("count")
    artifact_value = azure_artifacts_data.get("value")
    url_dict = {}
    for item in artifact_value:
        artifact_download_url = item.get("resource").get("downloadUrl")
        artifact_download_url = f"{artifact_download_url[:-3]}file&subPath=%2F"
        url_dict[item.get("name")] = artifact_download_url
    return url_dict


if __name__ == "__main__":
    main()
