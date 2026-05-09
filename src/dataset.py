import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class FakedditMultimodalDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, feature_extractor, max_len=64):
        """
        csv_file: Path to our filtered 'train_subset.csv'
        img_dir: 'images_sample' folder containing images
        tokenizer: BERT Tokenizer
        feature_extractor: ViT Feature Extractor
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = str(row['id'])
        text = str(row['clean_title'])
        label = int(row['2_way_label'])

        # --- TEXT PROCESSING (BERT) ---
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # --- IMAGE PROCESSING (ViT) ---
        # Find image (extension might be .jpg or .png, let's check)
        try:
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, f"{img_id}.png")
            
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.feature_extractor(image, return_tensors="pt")['pixel_values']
        except Exception as e:
            # If image is corrupted, try to get the next sample in the list
            print(f"Warning: Image {img_id} could not be read, skipping. Error: {e}")
            new_idx = (idx + 1) % len(self.data)
            return self.__getitem__(new_idx)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': pixel_values.squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }