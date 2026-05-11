import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTImageProcessor
import os

from src.dataset import FakedditMultimodalDataset
from src.models import MultimodalFusionModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hibrit Eğitim Cihazı: {device}")

    # RTX 3050 için hem metin hem görsel işlendiğinde VRAM'i korumak için batch size 8 veya 16 olmalı
    BATCH_SIZE = 8 
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # 1. İşlemcileri Yükle
    print("İşlemciler yükleniyor (BERT & ViT)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # 2. Veri Seti (Küçültülmüş 17k Subset)
    dataset = FakedditMultimodalDataset(
        csv_file='data/multimodel_subset.tsv', 
        img_dir='D:/463_project/data/images_sample/',
        tokenizer=tokenizer,
        feature_extractor=image_processor
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 3. Model ve Araçlar
    model = MultimodalFusionModel().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    print(f"MULTIMODAL FUSION Eğitimi Başlıyor: {len(dataset)} örnek...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Verileri çek ve cihaza taşı
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            
            # AMP ile Karışık Hassasiyetli Eğitim
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids, mask, pixel_values)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch: {epoch+1} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Bitti! Ortalama Loss: {avg_loss:.4f} ---")
        torch.save(model.state_dict(), f"multimodal_fusion_subset_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()