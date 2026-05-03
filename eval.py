import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset import FakedditMultimodalDataset
from src.models import MultimodalFakeNewsModel

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Test cihazı: {device}")

    # 1. Araçları ve Modeli Yükle
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    
    model = MultimodalFakeNewsModel()
    # En iyi epoch sonucunu yüklüyoruz
    model.load_state_dict(torch.load("multimodal_model_epoch_3.pt", map_location=device))
    model.to(device)
    model.eval()

    # 2. Yeni Validation Subset'i Yükle
    val_dataset = FakedditMultimodalDataset(
        csv_file='data/val_subset.csv',
        img_dir='D:/463_project/data/images_sample/',
        tokenizer=tokenizer,
        feature_extractor=image_processor
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    all_preds = []
    all_labels = []

    print(f"{len(val_dataset)} yeni örnek üzerinde model test ediliyor...")
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, mask, pixel_values)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Sonuçları Raporla
    print("\n" + "="*30)
    print("GÖRMEMİŞ VERİ PERFORMANSI")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=["Gerçek", "Sahte"]))
    print("\n--- HATA MATRİSİ ---")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()