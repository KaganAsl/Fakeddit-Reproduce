import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, ViTImageProcessor

from src.dataset import FakedditDataset
from src.models import MultimodalModelWithCrossAttention

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")

    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    JOINT_DIM = 768
    NUM_HEADS = 8
    DROPOUT = 0.1

    print("Models are loading...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    full_dataset = FakedditDataset(
        tsv_file='data/labels/multimodal.tsv',
        img_dir='data/subset/',
        tokenizer=tokenizer,
        feature_extractor=image_processor
    )

    total = len(full_dataset)
    train_size = int(0.9 * total)
    eval_size = total - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(full_dataset, [train_size, eval_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MultimodalModelWithCrossAttention(
        num_labels=2,
        joint_dim=JOINT_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Training starts with {len(train_dataset)} samples...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, mask, pixel_values)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch+1}/{EPOCHS} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Finished! Average Loss: {avg_loss:.4f} ---")

        torch.save(model.state_dict(), f"multimodal_attention_model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
