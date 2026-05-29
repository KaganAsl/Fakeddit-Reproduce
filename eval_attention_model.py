import argparse
import os

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, ViTImageProcessor

from src.dataset import FakedditMultimodalDataset
from src.models import MultimodalModelWithCrossAttention


def parse_args():
    p = argparse.ArgumentParser(description="Cross-Attention multimodal degerlendirme (2/3/6-way)")
    p.add_argument('--csv', default='data/multimodel_50k.tsv')
    p.add_argument('--img-dir', default='data/images_50k/')
    p.add_argument('--label-column', default='2_way_label',
                   choices=['2_way_label', '3_way_label', '6_way_label'])
    p.add_argument('--num-labels', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--output-prefix', default='attn_2way')
    p.add_argument('--epoch', type=int, default=3, help='Yuklenecek checkpoint epoch numarasi')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Task: {args.label_column} | num_labels={args.num_labels}")

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
    _, eval_dataset = random_split(full_dataset, [train_size, eval_size], generator=generator)

    test_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = MultimodalModelWithCrossAttention(
        num_labels=args.num_labels, joint_dim=768, num_heads=8, dropout=0.1,
    ).to(device)

    model_path = f"{args.output_prefix}_epoch_{args.epoch}.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Ensure you have trained the model first.")

    model.eval()

    all_preds, all_labels = [], []
    print("Multimodal Model with Cross Attention testing...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids, mask, pixel_values)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if all_labels:
        if args.num_labels == 2:
            target_names = ['Real', 'Fake']
        else:
            target_names = [f'class_{i}' for i in range(args.num_labels)]

        print("\n" + "=" * 50)
        print(f" RESULTS: CROSS ATTENTION ({args.label_column}) ")
        print("=" * 50)
        print(classification_report(all_labels, all_preds, labels=list(range(args.num_labels)),
                                    target_names=target_names, digits=4, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds, labels=list(range(args.num_labels))))


if __name__ == "__main__":
    main()
