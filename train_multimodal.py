import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTImageProcessor

from src.dataset import FakedditMultimodalDataset
from src.models import MultimodalFusionModel


def parse_args():
    p = argparse.ArgumentParser(description="Multimodal Fusion egitim (2/3/6-way)")
    p.add_argument('--csv', default='data/multimodel_50k.tsv')
    p.add_argument('--img-dir', default='data/images_50k/')
    p.add_argument('--label-column', default='2_way_label',
                   choices=['2_way_label', '3_way_label', '6_way_label'])
    p.add_argument('--num-labels', type=int, default=2)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--output-prefix', default='multimodal_fusion_2way')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hibrit Eğitim Cihazı: {device}")
    print(f"Task: {args.label_column} | num_labels={args.num_labels} | prefix={args.output_prefix}")

    # RTX 3050 için hem metin hem görsel işlendiğinde VRAM'i korumak için batch size 8 veya 16 olmalı

    # 1. İşlemcileri Yükle
    print("İşlemciler yükleniyor (BERT & ViT)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # 2. Veri Seti (Küçültülmüş 17k Subset)
    dataset = FakedditMultimodalDataset(
        csv_file=args.csv,
        img_dir=args.img_dir,
        tokenizer=tokenizer,
        feature_extractor=image_processor,
        label_column=args.label_column,
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Sınıf dengesizliği için ağırlıklı loss (özellikle 3/6-way nadir sınıflar)
    labels_all = pd.read_csv(args.csv, sep='\t')[args.label_column].to_numpy()
    classes = np.arange(args.num_labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels_all)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.round(3).tolist()}")

    # 3. Model ve Araçlar
    model = MultimodalFusionModel(num_classes=args.num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    scaler = torch.amp.GradScaler('cuda')

    print(f"MULTIMODAL FUSION Eğitimi Başlıyor: {len(dataset)} örnek...")
    
    for epoch in range(args.epochs):
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
                print(f"Epoch: {epoch+1}/{args.epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Bitti! Ortalama Loss: {avg_loss:.4f} ---")
        torch.save(model.state_dict(), f"{args.output_prefix}_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
