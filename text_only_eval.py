import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Kendi modüllerin
from src.dataset import FakedditMultimodalDataset
from src.models import TextOnlyFakeNewsModel

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # 1. Hazırlık
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Test verisini yükle (Eğer ayrı bir test setin yoksa aynı dosyayı kullanabilirsin)
    dataset = FakedditMultimodalDataset(
        csv_file='data/multimodel.tsv', 
        img_dir='D:/463_project/data/images_sample/',
        tokenizer=tokenizer,
        feature_extractor=None
    )
    
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # 2. Model ve Ağırlıkları Yükleme
    model = TextOnlyFakeNewsModel().to(device)
    model.load_state_dict(torch.load('text_only_model_epoch_3.pt'))
    model.eval()

    all_preds = []
    all_labels = []

    print("Model test ediliyor, bu işlem çok kısa sürecek...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, mask)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Raporu Yazdırma
    print("\n" + "="*30)
    print(" BASELINE 1 (TEXT-ONLY) SONUÇLARI ")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=['Gerçek', 'Sahte']))
    
    print("\nHata Matrisi (Confusion Matrix):")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()