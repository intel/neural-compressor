import os
import sys


def main(inputs):
    # output_file = f"{input_file}.new"
    for input_file in inputs:
        output_file = input_file
        res = []
        with open(input_file, "r") as f:
            for line in f.readlines():
                start = 0
                while True:
                    pos = line.find('href="./', start)

                    if pos >= 0:
                        # print("find", line)
                        end_pos = line.find('">', pos)
                        line = line[:end_pos] + '/README.md">' + line[end_pos + 2 :]
                        start = end_pos + 2
                    else:
                        break

                res.append(line)

        with open(output_file, "w") as f:
            f.write("".join(res))

            print(f"save to {output_file}")


if __name__ == "__main__":
    main(sys.argv[1:])
