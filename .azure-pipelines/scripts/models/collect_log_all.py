import argparse
import os

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--logs_dir", type=str, default=".")
parser.add_argument("--output_dir", type=str, default=".")
args = parser.parse_args()
print(args)


def main():
    file_dir = args.logs_dir
    summary_content = ['OS;Platform;Framework;Version;Precision;Model;Mode;Type;BS;Value;Url\n']
    tuning_info_content = ['OS;Platform;Framework;Version;Model;Strategy;Tune_time\n']
    # get full path of all files
    for root, dirs, files in os.walk(file_dir):
        for name in files:
            file_name = os.path.join(root, name)
            print(file_name)
            if '_summary.log' in name:
                for line in open(file_name, "r"):
                    # print(line)
                    if 'linux' in line:
                        summary_content.append(line)
            if '_tuning_info.log' in name:
                for line in open(file_name, "r"):
                    # print(line)
                    if 'linux' in line:
                        tuning_info_content.append(line)
    f = open(args.output_dir + '/summary.log', "a")
    for summary in summary_content:
        f.writelines(str(summary))
    f2 = open(args.output_dir + '/tuning_info.log', "a")
    for tuning_info in tuning_info_content:
        f2.writelines(str(tuning_info))


if __name__ == '__main__':
    main()
