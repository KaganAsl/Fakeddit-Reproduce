import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTImageProcessor
import os

# Import our own modules
from src.dataset import FakedditMultimodalDataset
from src.models import MultimodalFakeNewsModel

def main():
    # 1. Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")

    # 2. Parameters
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # 3. Preparation (Tokenizer & Processor)
    print("Models are loading...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

    # 4. Data Loaders
    dataset = FakedditMultimodalDataset(
        csv_file='data/train_subset.csv',
        img_dir='D:/463_project/data/images_sample/',
        tokenizer=tokenizer,
        feature_extractor=image_processor
    )
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Model, Loss and Optimizer
    model = MultimodalFakeNewsModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 6. TRAINING LOOP
    print(f"Training starts with {len(dataset)} samples...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, mask, pixel_values)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch+1}/{EPOCHS} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Finished! Average Loss: {avg_loss:.4f} ---")

        # Save model at the end of each epoch
        torch.save(model.state_dict(), f"multimodal_model_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()