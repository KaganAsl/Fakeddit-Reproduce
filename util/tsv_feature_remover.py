from pathlib import Path
import pandas as pd
import sys

input_file = sys.argv[1]

output_file = sys.argv[2]

columns_to_remove = [
    "author",
    "created_utc",
    "domain",
    "hasImage",
    "image_url",
    "linked_submission_id",
    "num_comments",
    "score",
    "subreddit",
    "title",
    "upvote_ratio"
]

df = pd.read_csv(input_file, sep="\t")

df = df.drop(columns=columns_to_remove, errors="ignore")

df.to_csv(output_file, sep="\t", index=False)

print(f"Saved filtered file to: {output_file}")