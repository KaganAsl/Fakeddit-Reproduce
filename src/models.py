import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
from attention import CrossAttention

# 1. Baseline 1: Sadece Metin Modeli
class TextOnlyFakeNewsModel(nn.Module):
    def __init__(self, num_classes=2):
        super(TextOnlyFakeNewsModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # BERT'in [CLS] token çıkışını kullanıyoruz
        return self.classifier(outputs.pooler_output)

# 2. Baseline 2: Sadece Görsel Modeli
class ImageOnlyFakeNewsModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageOnlyFakeNewsModel, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        # ViT'in [CLS] token çıkışını kullanıyoruz
        return self.classifier(outputs.pooler_output)

# 3. FINAL MODEL: Multimodal Fusion (BERT + ViT)
class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MultimodalFusionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # BERT (768) + ViT (768) birleşimi = 1536
        combined_features_dim = self.bert.config.hidden_size + self.vit.config.hidden_size
        
        # Hata buradaydı: İsim 'classifier' olarak güncellendi
        self.classifier = nn.Sequential(
            nn.Linear(combined_features_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # Metin özelliklerini çıkar
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output 
        
        # Görsel özelliklerini çıkar
        image_outputs = self.vit(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output 
        
        # İki vektörü yan yana ekle
        combined_features = torch.cat((text_features, image_features), dim=1)
        
        # Birleştirilmiş vektörü sınıflandırıcıya gönder
        return self.classifier(combined_features)

# 4. Cross Attention
class MultimodalModelWithCrossAttention(nn.Module):
    def __init__(self, num_labels=2, joint_dim=768, num_heads=8, dropout=0.1):
        super(MultimodalModelWithCrossAttention, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        self.cross_attention = CrossAttention(
            vit_dim=768,
            bert_dim=768,
            joint_dim=joint_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(joint_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output

        image_outputs = self.vit(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output

        fused_img, fused_txt = self.cross_attention(image_features, text_features)

        combined = torch.cat((fused_img, fused_txt), dim=1)
        logits = self.classifier(combined)
        return logits