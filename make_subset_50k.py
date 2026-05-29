import pandas as pd
from sklearn.model_selection import train_test_split

# Subset III: 50K dengeli (stratified) alt-kume.
# Subset I (multimodel.tsv, 176.363 satir) ile Subset II (%10, 17.636 satir) arasi bir ara olcek.
INPUT = 'data/multimodel.tsv'
OUTPUT = 'data/multimodel_50k.tsv'
TARGET_SIZE = 50000

df = pd.read_csv(INPUT, sep='\t')

# Etiket oranini koruyarak 50.000 satir sec
subset, _ = train_test_split(
    df,
    train_size=TARGET_SIZE,
    stratify=df['2_way_label'],
    random_state=42,
)

subset.to_csv(OUTPUT, sep='\t', index=False)

dist = subset['2_way_label'].value_counts().sort_index()
total = len(subset)
print(f"Yeni veri seti hazir: {total} satir -> {OUTPUT}")
for label, count in dist.items():
    name = 'Gercek(0)' if label == 0 else 'Sahte(1)'
    print(f"  {name}: {count} ({count / total:.1%})")
