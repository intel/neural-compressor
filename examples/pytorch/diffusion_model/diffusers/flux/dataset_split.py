import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--split_num', type=int, required=True)
parser.add_argument('--limit', default=-1, type=int)
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', default="subset", type=str)
args = parser.parse_args()

# load the TSV file
df = pd.read_csv(args.input_file, sep='\t')

if args.limit > 0:
    df = df.iloc[0:args.limit]

if args.split_num <= 0:
    parser.error("--split_num must be greater than 0")

subset_size, remainder = divmod(len(df), args.split_num)
for i in range(args.split_num):
    start = i * subset_size + min(i, remainder)
    end = start + subset_size + (1 if i < remainder else 0)
    df_subset = df.iloc[start:end]
    df_subset.to_csv(f"{args.output_file}_{i}.tsv", sep='\t', index=False)
