import pandas as pd
import os

# Konfigürasyon
IMAGE_FOLDER = "D:/463_project/data/images_sample/" # Görsellerin olduğu klasör
TSV_PATH = "data/multimodal_train.tsv"             # v2.0'dan indirdiğin ana dosya
OUTPUT_PATH = "data/train_subset.csv"              # Oluşacak yeni küçük dosya

def prepare_data():
    # 1. Klasördeki görsel ID'lerini topla
    print("Görsel listesi okunuyor...")
    extracted_ids = [f.split('.')[0] for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Klasörde {len(extracted_ids)} görsel bulundu.")

    # 2. Ana TSV dosyasını oku (Sadece gerekli sütunları yükleyerek RAM'i koruyalım)
    print("Ana veri dosyası yükleniyor (Bu biraz vakit alabilir)...")
    df_full = pd.read_csv(TSV_PATH, sep='\t', usecols=['id', 'clean_title', '2_way_label'])

    # 3. Filtreleme yap
    df_subset = df_full[df_full['id'].isin(extracted_ids)]
    
    # 4. Kaydet
    df_subset.to_csv(OUTPUT_PATH, index=False)
    print(f"Başarılı! {len(df_subset)} eşleşen satır '{OUTPUT_PATH}' dosyasına kaydedildi.")

if __name__ == "__main__":
    prepare_data()