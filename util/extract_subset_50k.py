import os
import zipfile
import pandas as pd

# Subset III icin gereken ~50K gorseli C:\images_subset.zip'ten cikar.
# Tum 200K yerine sadece TSV'deki id'lere karsilik gelen {id}.jpg dosyalari alinir.
ARCHIVE = r"C:\images_subset.zip"
TSV = "data/multimodel_50k.tsv"
OUTPUT_DIR = "data/images_50k/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(TSV, sep='\t')
wanted = {f"{img_id}.jpg" for img_id in df['id'].astype(str)}
print(f"Aranan gorsel sayisi (TSV): {len(wanted)}")

extracted = 0
missing = 0
with zipfile.ZipFile(ARCHIVE, 'r') as zf:
    for info in zf.infolist():
        if info.is_dir():
            continue
        base = os.path.basename(info.filename)
        if base in wanted:
            target = os.path.join(OUTPUT_DIR, base)
            with zf.open(info) as src, open(target, 'wb') as dst:
                dst.write(src.read())
            extracted += 1
            wanted.discard(base)
            if extracted % 2000 == 0:
                print(f"{extracted} gorsel cikarildi...")

missing = len(wanted)
print(f"Islem tamam! {extracted} gorsel '{OUTPUT_DIR}' klasorune cikarildi.")
print(f"Arsivde bulunamayan id sayisi: {missing}")
