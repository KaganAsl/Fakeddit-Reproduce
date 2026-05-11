import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
import os

from src.dataset import FakedditMultimodalDataset
from src.models import ImageOnlyFakeNewsModel

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Görsel Eğitimi Cihazı: {device}")

    # 🚀 ÖNEMLİ: RTX 3050 için Batch Size 16 veya 32 idealdir
    BATCH_SIZE = 16 
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    print("ViT Processor yükleniyor...")
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    dataset = FakedditMultimodalDataset(
        csv_file='data/multimodel_subset.tsv', 
        img_dir='D:/463_project/data/images_sample/',
        tokenizer=None, # Metne ihtiyacımız yok
        feature_extractor=image_processor
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4, # Görsel işleme için CPU yükü artar
        pin_memory=True
    )

    model = ImageOnlyFakeNewsModel().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    print(f"Görsel Modeli (Baseline 2) Eğitimi Başlıyor: {len(dataset)} örnek...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Sadece görselleri alıyoruz
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(pixel_values)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch: {epoch+1} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"--- Epoch {epoch+1} Tamamlandı! Ortalama Loss: {epoch_loss/len(train_loader):.4f} ---")
        torch.save(model.state_dict(), f"image_only_subset_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()