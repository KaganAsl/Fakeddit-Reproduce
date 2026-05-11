import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import os

# Kendi modüllerimiz
from src.dataset import FakedditMultimodalDataset
from src.models import TextOnlyFakeNewsModel

def main():
    # 1. Cihaz Kurulumu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hızlı Eğitim Cihazı: {device}")

    # 2. Hiperparametreler
    # Metin modunda bellek kullanımı çok düşüktür. 
    # Eğer GPU belleğin (VRAM) yeterliyse 64 hatta 128 yapabilirsin.
    BATCH_SIZE = 64 
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # 3. Tokenizer Yükleme
    print("BERT Tokenizer yükleniyor...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 4. Veri Seti Hazırlığı
    # OPTİMİZASYON: feature_extractor=None yaparak görsel işlemlerini tamamen bypass ediyoruz.
    # Dosya adını senin temizlediğin dosya ismiyle (örn: data/multimodal_fixed.tsv) güncellemelisin.
    dataset = FakedditMultimodalDataset(
        csv_file='data/multimodel.tsv', 
        img_dir='D:/463_project/data/images_sample/',
        tokenizer=tokenizer,
        feature_extractor=None 
    )
    
    # 5. Veri Yükleyici (DataLoader) Ayarları
    # num_workers: İşlemci çekirdek sayına göre 4, 8 veya 12 yapabilirsin.
    train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=8, 
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 6. Model ve Optimizer
    model = TextOnlyFakeNewsModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\n🚀 HIZLI METİN EĞİTİMİ (BASELINE 1) BAŞLIYOR")
    print(f"Toplam Örnek: {len(dataset)}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Görsel İşlemleri: ATLANARAK GEÇİLİYOR\n")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, mask)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}/{EPOCHS} | Progress: %{100 * batch_idx / len(train_loader):.1f} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n--- Epoch {epoch+1} Tamamlandı! Ortalama Kayıp: {avg_loss:.4f} ---\n")
        
        # Her epoch sonunda ağırlıkları kaydet
        torch.save(model.state_dict(), f"text_only_model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()