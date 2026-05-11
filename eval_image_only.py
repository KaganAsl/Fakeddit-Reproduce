import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Kendi modüllerimiz
from src.dataset import FakedditMultimodalDataset
from src.models import ImageOnlyFakeNewsModel

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # 1. Hazırlık
    print("ViT Processor yükleniyor...")
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    # Test verisini yükle (Subset dosyasını kullanıyoruz)
    dataset = FakedditMultimodalDataset(
        csv_file='data/multimodel_subset.tsv', 
        img_dir='D:/463_project/data/images_sample/',
        tokenizer=None,
        feature_extractor=image_processor
    )
    
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 2. Model ve Ağırlıkları Yükleme
    model = ImageOnlyFakeNewsModel().to(device)
    # En son kaydedilen ağırlığı yükle
    model.load_state_dict(torch.load('image_only_subset_epoch_3.pt'))
    model.eval()

    all_preds = []
    all_labels = []

    print("Görsel modeli test ediliyor...")
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            outputs = model(pixel_values)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Raporu Yazdırma
    print("\n" + "="*30)
    print(" BASELINE 2 (IMAGE-ONLY) SONUÇLARI ")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=['Gerçek', 'Sahte']))
    
    print("\nHata Matrisi (Confusion Matrix):")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()