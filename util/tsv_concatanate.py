from pathlib import Path
import pandas as pd
import sys

input_folder = sys.argv[1]
output_file = sys.argv[2]

tsv_files = list(Path(input_folder).glob("*.tsv"))

if not tsv_files:
    raise FileNotFoundError("No TSV files found.")

dfs = []
for file in tsv_files:
    print(f"Reading: {file}")
    df = pd.read_csv(file, sep="\t")
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

combined_df.to_csv(output_file, sep="\t", index=False)

print(f"Combined {len(tsv_files)} files into: {output_file}")