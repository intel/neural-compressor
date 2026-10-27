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

"""Create a DrawBench TSV (id, caption) for Z-Image evaluation.

The ids are 0..N-1 in dataset order, matching the seeds used by
``main.py`` (seed + row index) and by autoquant-agent's diffusion harness.
"""

import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="sayakpaul/drawbench")
parser.add_argument("--split", default="train")
parser.add_argument("--prompt_field", default="Prompts")
parser.add_argument("--output", default="drawbench.tsv")
args = parser.parse_args()


def main():
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    rows = [{"id": i, "caption": row[args.prompt_field]} for i, row in enumerate(ds)]
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"wrote {len(rows)} prompts to {args.output}")


if __name__ == "__main__":
    main()
