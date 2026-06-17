import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

from src.dataset import FakedditMultimodalDataset
from src.models import TextOnlyFakeNewsModel


def parse_args():
    p = argparse.ArgumentParser(description="Text-Only training (2/3/6-way)")
    p.add_argument('--csv', default='data/multimodel_50k.tsv')
    p.add_argument('--img-dir', default='data/images_50k/')
    p.add_argument('--label-column', default='2_way_label',
                   choices=['2_way_label', '3_way_label', '6_way_label'])
    p.add_argument('--num-labels', type=int, default=2)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--output-prefix', default='text_only_2way')
    p.add_argument('--split-size', type=float, default=0.7)
    p.add_argument('--loss-csv', default='text_only_batch_losses.csv', help='Path to save batch losses')
    p.add_argument('--checkpoint', default=None, help='Path to a .pt model file to resume training from')
    p.add_argument('--start-epoch', type=int, default=0, help='Epoch to start from (e.g., 3 if you already trained 3 epochs)')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")
    print(f"Task: {args.label_column} | num_labels={args.num_labels} | prefix={args.output_prefix}")

    print("Loading BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    full_dataset = FakedditMultimodalDataset(
        csv_file=args.csv,
        img_dir=args.img_dir,
        tokenizer=tokenizer,
        feature_extractor=None,
        label_column=args.label_column,
    )

    total = len(full_dataset)
    train_size = int(args.split_size * total)
    eval_size = total - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    labels_all = pd.read_csv(args.csv, sep='\t')[args.label_column].to_numpy()
    classes = np.arange(args.num_labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels_all)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights.round(3).tolist()}")

    model = TextOnlyFakeNewsModel(num_classes=args.num_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    print(f"\nTEXT-ONLY (BASELINE 1) TRAINING")
    print(f"Training starts: {len(train_dataset)} train / {len(eval_dataset)} eval samples")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image processing: SKIPPED\n")

    for epoch in range(args.start_epoch, args.epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch: {epoch+1}/{args.epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                with open(args.loss_csv, "a") as f:
                    f.write(f"{args.output_prefix},{epoch+1},{loss.item():.4f}\n")

        avg_train_loss = epoch_loss / len(train_loader)

        # --- Evaluation ---
        model.eval()
        eval_loss = 0
        correct = 0
        total_eval = 0

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, mask)
                loss = criterion(outputs, labels)

                eval_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total_eval += labels.size(0)

        avg_eval_loss = eval_loss / len(eval_loader)
        eval_acc = correct / total_eval

        print(f"\n--- Epoch {epoch+1} Done! Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | Eval Acc: {eval_acc:.4f} ---\n")
        torch.save(model.state_dict(), f"{args.output_prefix}_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
