import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import ViTImageProcessor

from src.dataset import FakedditMultimodalDataset
from src.models import ImageOnlyFakeNewsModel


def parse_args():
    p = argparse.ArgumentParser(description="Image-Only training (2/3/6-way)")
    p.add_argument('--csv', default='data/multimodel_50k.tsv')
    p.add_argument('--img-dir', default='data/images_50k/')
    p.add_argument('--label-column', default='2_way_label',
                   choices=['2_way_label', '3_way_label', '6_way_label'])
    p.add_argument('--num-labels', type=int, default=2)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--output-prefix', default='image_only_2way')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")
    print(f"Task: {args.label_column} | num_labels={args.num_labels} | prefix={args.output_prefix}")

    print("Loading ViT Processor...")
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    full_dataset = FakedditMultimodalDataset(
        csv_file=args.csv,
        img_dir=args.img_dir,
        tokenizer=None,
        feature_extractor=image_processor,
        label_column=args.label_column,
    )

    # 90/10 train/eval split
    total = len(full_dataset)
    train_size = int(0.9 * total)
    eval_size = total - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    labels_all = pd.read_csv(args.csv, sep='\t')[args.label_column].to_numpy()
    classes = np.arange(args.num_labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels_all)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.round(3).tolist()}")

    model = ImageOnlyFakeNewsModel(num_classes=args.num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    scaler = torch.amp.GradScaler('cuda')

    print(f"Training starts: {len(train_dataset)} train / {len(eval_dataset)} eval samples")

    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
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
                print(f"Epoch: {epoch+1}/{args.epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # --- Evaluation ---
        model.eval()
        eval_loss = 0
        correct = 0
        total_eval = 0

        with torch.no_grad():
            for batch in eval_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(pixel_values)
                    loss = criterion(outputs, labels)

                eval_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total_eval += labels.size(0)

        avg_eval_loss = eval_loss / len(eval_loader)
        eval_acc = correct / total_eval

        print(f"--- Epoch {epoch+1} Done! Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | Eval Acc: {eval_acc:.4f} ---")
        torch.save(model.state_dict(), f"{args.output_prefix}_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()