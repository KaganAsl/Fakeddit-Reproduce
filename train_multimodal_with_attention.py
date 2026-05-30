import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, ViTImageProcessor

from src.dataset import FakedditMultimodalDataset
from src.models import MultimodalModelWithCrossAttention, FUSION_METHODS


def parse_args():
    p = argparse.ArgumentParser(description="Cross-Attention multimodal egitim (2/3/6-way)")
    p.add_argument('--csv', default='data/multimodel_50k.tsv')
    p.add_argument('--img-dir', default='data/images_50k/')
    p.add_argument('--label-column', default='2_way_label',
                   choices=['2_way_label', '3_way_label', '6_way_label'])
    p.add_argument('--num-labels', type=int, default=2)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--output-prefix', default='attn_2way')
    p.add_argument('--fusion', default='concat', choices=FUSION_METHODS,
                   help='Embedding fusion method: concat|add|max|average|multiply')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")
    print(f"Task: {args.label_column} | num_labels={args.num_labels} | fusion={args.fusion} | prefix={args.output_prefix}")

    JOINT_DIM, NUM_HEADS, DROPOUT = 768, 8, 0.1

    print("Models are loading...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    full_dataset = FakedditMultimodalDataset(
        csv_file=args.csv,
        img_dir=args.img_dir,
        tokenizer=tokenizer,
        feature_extractor=image_processor,
        label_column=args.label_column,
    )

    total = len(full_dataset)
    train_size = int(0.9 * total)
    eval_size = total - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(full_dataset, [train_size, eval_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Sinif dengesizligi icin agirlikli loss (ozellikle 3/6-way nadir siniflar)
    labels_all = pd.read_csv(args.csv, sep='\t')[args.label_column].to_numpy()
    classes = np.arange(args.num_labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels_all)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.round(3).tolist()}")

    model = MultimodalModelWithCrossAttention(
        num_labels=args.num_labels,
        joint_dim=JOINT_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        fusion=args.fusion,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')

    print(f"Training starts with {len(train_dataset)} samples...")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
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
        print(f"--- Epoch {epoch+1} Finished! Average Loss: {avg_loss:.4f} ---")
        torch.save(model.state_dict(), f"{args.output_prefix}_epoch_{epoch+1}.pt")


if __name__ == "__main__":
    main()
