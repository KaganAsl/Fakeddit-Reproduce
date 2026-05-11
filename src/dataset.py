import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class FakedditMultimodalDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer=None, feature_extractor=None, max_len=128):
        self.data = pd.read_csv(csv_file, sep='\t')
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. Metin İşleme (Sadece tokenizer varsa çalışır)
        text = str(row['clean_title'])
        sample = {}
        
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            sample['input_ids'] = encoding['input_ids'].flatten()
            sample['attention_mask'] = encoding['attention_mask'].flatten()

        # 2. Görsel İşleme (Sadece feature_extractor varsa çalışır)
        # Görsel ismini oluştur (Fakeddit yapısına göre id_ ile başlar)
        img_id = row['id']
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        
        if self.feature_extractor is not None:
            try:
                # OPTİMİZASYON: Şeffaf (RGBA) veya paletli görselleri RGB'ye çevir
                image = Image.open(img_path).convert('RGB')
                pixel_values = self.feature_extractor(images=image, return_tensors="pt")['pixel_values']
                sample['pixel_values'] = pixel_values.squeeze()
            except Exception as e:
                # Görsel bulunamazsa veya bozuksa boş/sıfır bir tensör dön (Eğitimi durdurmaz)
                # print(f"Hata: {img_path} yüklenemedi. Boş görsel dönülüyor.")
                sample['pixel_values'] = torch.zeros(3, 224, 224)

        # 3. Etiket (2_way_label: 0=Gerçek, 1=Sahte)
        sample['label'] = torch.tensor(row['2_way_label'], dtype=torch.long)
        
        return sample