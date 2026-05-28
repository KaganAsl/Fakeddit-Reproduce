import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, ViTImageProcessor
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import FakedditDataset
from src.models import MultimodalModelWithCrossAttention

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

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
    _, eval_dataset = random_split(full_dataset, [train_size, eval_size], generator=generator)
    
    test_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

    JOINT_DIM = 768
    NUM_HEADS = 8
    DROPOUT = 0.1

    model = MultimodalModelWithCrossAttention(
        num_labels=2,
        joint_dim=JOINT_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(device)
    
    # Check for the latest epoch, or default to epoch 3
    import os
    model_path = 'multimodal_attention_model_epoch_3.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Ensure you have trained the model first.")
    
    model.eval()

    all_preds = []
    all_labels = []

    print("Multimodal Model with Cross Attention testing...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, mask, pixel_values)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if len(all_labels) > 0:
        print("\n" + "="*45)
        print(" FINAL RESULTS: MULTIMODAL CROSS ATTENTION ")
        print("="*45)
        print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
        
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()
