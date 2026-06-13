import argparse

import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import FakedditMultimodalDataset
from src.models import TextOnlyFakeNewsModel

# Class names for each task
LABEL_NAMES = {
    '2_way_label': ['Real', 'Fake'],
    '3_way_label': ['Real', 'Fake-Text', 'Fake-Image'],
    '6_way_label': ['Real', 'Satire', 'Misleading Content',
                    'False Connection', 'Manipulated', 'Fabricated'],
}


def parse_args():
    p = argparse.ArgumentParser(description="Text-Only eval (2/3/6-way)")
    p.add_argument('--csv', default='data/multimodel_50k.tsv')
    p.add_argument('--img-dir', default='data/images_50k/')
    p.add_argument('--label-column', default='2_way_label',
                   choices=['2_way_label', '3_way_label', '6_way_label'])
    p.add_argument('--num-labels', type=int, default=2)
    p.add_argument('--model-path', default='text_only_2way_epoch_3.pt',
                   help='Saved model weights file')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--split-size', type=float, default=0.7)
    return p.parse_args()


def evaluate():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Task: {args.label_column} | num_labels={args.num_labels} | model={args.model_path}")

    # 1. Preparation
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
    _, eval_dataset = random_split(full_dataset, [train_size, eval_size], generator=generator)

    print(f"Evaluating on {len(eval_dataset)} samples (10% held-out split)")
    test_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Load Model
    model = TextOnlyFakeNewsModel(num_classes=args.num_labels).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating Text-Only model (BERT)...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Print Results
    target_names = LABEL_NAMES.get(args.label_column, [str(i) for i in range(args.num_labels)])
    print("\n" + "="*35)
    print(" FINAL RESULTS: TEXT-ONLY (BASELINE 1) ")
    print("="*35)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=3))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()