import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTImageProcessor
import os

# Kendi yazdığımız modülleri içe aktaralım
from src.dataset import FakedditMultimodalDataset
from src.models import MultimodalFakeNewsModel

def main():
    # 1. Cihaz Ayarı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eğitim cihazı: {device}")

    # 2. Parametreler
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # 3. Hazırlık (Tokenizer & Processor)
    print("Modeller yükleniyor...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # 4. Veri Yükleyiciler
    dataset = FakedditMultimodalDataset(
        csv_file='data/train_subset.csv',
        img_dir='D:/463_project/data/images_sample/',
        tokenizer=tokenizer,
        feature_extractor=image_processor
    )
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Model, Loss ve Optimizer
    model = MultimodalFakeNewsModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 6. EĞİTİM DÖNGÜSÜ
    print(f"{len(dataset)} örnek ile eğitim başlıyor...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Verileri cihaza taşı
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            # İleri besleme (Forward)
            optimizer.zero_grad()
            outputs = model(input_ids, mask, pixel_values)
            loss = criterion(outputs, labels)

            # Geri besleme (Backward)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch+1}/{EPOCHS} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Bitti! Ortalama Loss: {avg_loss:.4f} ---")

        # Her epoch sonunda modeli kaydet
        torch.save(model.state_dict(), f"multimodal_model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()