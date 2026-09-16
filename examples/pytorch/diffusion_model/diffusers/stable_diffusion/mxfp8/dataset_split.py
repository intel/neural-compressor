# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import pandas as pd


def split_dataset(input_file, output_prefix, split_num, limit=-1):
    if split_num <= 0:
        raise ValueError("split_num must be greater than zero")

    dataframe = pd.read_csv(input_file, sep="\t")
    if limit > 0:
        dataframe = dataframe.iloc[:limit]

    subset_size, remainder = divmod(len(dataframe), split_num)
    for index in range(split_num):
        start = index * subset_size + min(index, remainder)
        end = start + subset_size + (1 if index < remainder else 0)
        dataframe.iloc[start:end].to_csv(f"{output_prefix}_{index}.tsv", sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description="Split a TSV dataset into balanced subsets.")
    parser.add_argument("--split_num", type=int, required=True)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", default="subset")
    args = parser.parse_args()
    split_dataset(args.input_file, args.output_file, args.split_num, args.limit)


if __name__ == "__main__":
    main()
