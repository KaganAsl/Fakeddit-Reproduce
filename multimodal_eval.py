import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Kendi modüllerimiz
from src.dataset import FakedditMultimodalDataset
from src.models import MultimodalFusionModel

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # 1. Hazırlık
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    # Test verisini yükle (Tüm verideki genel başarıyı görmek için orijinal dosyayı kullanabilirsin)
    dataset = FakedditMultimodalDataset(
        csv_file='data/multimodel_subset.tsv', 
        img_dir='D:/463_project/data/images_sample/',
        tokenizer=tokenizer,
        feature_extractor=image_processor
    )
    
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    # 2. Model Yükleme
    model = MultimodalFusionModel().to(device)
    model.load_state_dict(torch.load('multimodal_fusion_subset_epoch_3.pt'))
    model.eval()

    all_preds = []
    all_labels = []

    print("Multimodal Fusion modeli test ediliyor (BERT + ViT)...")
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

    # 3. Sonuçları Yazdır
    print("\n" + "="*35)
    print(" FINAL RESULTS: MULTIMODAL FUSION ")
    print("="*35)
    print(classification_report(all_labels, all_preds, target_names=['Gerçek', 'Sahte']))
    
    print("\nHata Matrisi (Confusion Matrix):")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()