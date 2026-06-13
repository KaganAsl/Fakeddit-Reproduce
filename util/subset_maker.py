import pandas as pd
from sklearn.model_selection import train_test_split

# Orijinal temizlenmiş dosyayı oku
df = pd.read_csv('data/multimodel.tsv', sep='\t')

# %10'luk (yaklaşık 17.600 satır) dengeli bir alt küme al
subset, _ = train_test_split(
    df, 
    test_size=0.9, 
    stratify=df['2_way_label'], # Etiket oranını korur
    random_state=42
)

# Yeni dosyayı kaydet
subset.to_csv('data/multimodel_subset.tsv', sep='\t', index=False)
print(f"Yeni veri seti hazır: {len(subset)} satır.")