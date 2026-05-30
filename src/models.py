import torch
import torch.nn as nn
from transformers import BertModel, ViTModel
from .attention import CrossAttention

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
FUSION_METHODS = ('concat', 'add', 'max', 'average', 'multiply')

class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=2, fusion='concat'):
        super(MultimodalFusionModel, self).__init__()
        assert fusion in FUSION_METHODS, f"fusion must be one of {FUSION_METHODS}, got '{fusion}'"
        self.fusion = fusion

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        # concat -> 768+768=1536,  element-wise ops -> 768
        if fusion == 'concat':
            combined_features_dim = self.bert.config.hidden_size + self.vit.config.hidden_size
        else:
            combined_features_dim = self.bert.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(combined_features_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def _fuse(self, text_features, image_features):
        if self.fusion == 'concat':
            return torch.cat((text_features, image_features), dim=1)
        elif self.fusion == 'add':
            return text_features + image_features
        elif self.fusion == 'max':
            return torch.max(text_features, image_features)
        elif self.fusion == 'average':
            return (text_features + image_features) / 2.0
        elif self.fusion == 'multiply':
            return text_features * image_features

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output

        image_outputs = self.vit(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output

        combined_features = self._fuse(text_features, image_features)
        return self.classifier(combined_features)

# 4. Cross Attention
class MultimodalModelWithCrossAttention(nn.Module):
    def __init__(self, num_labels=2, joint_dim=768, num_heads=8, dropout=0.1, fusion='concat'):
        super(MultimodalModelWithCrossAttention, self).__init__()
        assert fusion in FUSION_METHODS, f"fusion must be one of {FUSION_METHODS}, got '{fusion}'"
        self.fusion = fusion

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

        self.cross_attention = CrossAttention(
            vit_dim=768,
            bert_dim=768,
            joint_dim=joint_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        classifier_in = joint_dim * 2 if fusion == 'concat' else joint_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def _fuse(self, feat_a, feat_b):
        if self.fusion == 'concat':
            return torch.cat((feat_a, feat_b), dim=1)
        elif self.fusion == 'add':
            return feat_a + feat_b
        elif self.fusion == 'max':
            return torch.max(feat_a, feat_b)
        elif self.fusion == 'average':
            return (feat_a + feat_b) / 2.0
        elif self.fusion == 'multiply':
            return feat_a * feat_b

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output

        image_outputs = self.vit(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output

        fused_img, fused_txt = self.cross_attention(image_features, text_features)

        combined = self._fuse(fused_img, fused_txt)
        logits = self.classifier(combined)
        return logits