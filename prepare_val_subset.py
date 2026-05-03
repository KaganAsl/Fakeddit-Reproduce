import pandas as pd
import os

# Konfigürasyon
IMAGE_FOLDER = "D:/463_project/data/images_sample/" # Görsellerin olduğu klasör
VAL_TSV_PATH = "data/multimodal_validate.tsv"       # v2.0 doğrulama dosyası
OUTPUT_PATH = "data/val_subset.csv"                 # Oluşacak dosya

def prepare_val_data():
    print("Görsel listesi kontrol ediliyor...")
    extracted_ids = [f.split('.')[0] for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print("Doğrulama dosyası yükleniyor...")
    # Sadece gerekli sütunları alarak RAM tasarrufu yapalım
    df_val = pd.read_csv(VAL_TSV_PATH, sep='\t', usecols=['id', 'clean_title', '2_way_label'])

    print("Eşleştirme yapılıyor...")
    df_val_subset = df_val[df_val['id'].astype(str).isin(extracted_ids)]
    
    df_val_subset.to_csv(OUTPUT_PATH, index=False)
    print(f"Başarılı! {len(df_val_subset)} test örneği '{OUTPUT_PATH}' dosyasına kaydedildi.")

if __name__ == "__main__":
    prepare_val_data()