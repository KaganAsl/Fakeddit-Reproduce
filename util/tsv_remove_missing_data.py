import pandas as pd
import os
import sys

tsv_path = sys.argv[1]
folder_path = sys.argv[2]
output_tsv = sys.argv[3]
id_column = 'id'

df = pd.read_csv(tsv_path, sep='\t')

existing_files = {os.path.splitext(f)[0] for f in os.listdir(folder_path)}

df_filtered = df[df[id_column].astype(str).isin(existing_files)]

df_filtered.to_csv(output_tsv, sep='\t', index=False)

print(f"Original rows: {len(df)}")
print(f"Rows remaining: {len(df_filtered)}")