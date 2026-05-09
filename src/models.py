import torch
import torch.nn as nn
from transformers import BertModel, ViTModel

class MultimodalFakeNewsModel(nn.Module):
    def __init__(self, num_labels=2):
        super(MultimodalFakeNewsModel, self).__init__()
        # Load Language and Image experts
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # BERT (768) + ViT (768) = 1536 total features
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, pixel_values):
        # 1. Text Features
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output # [batch, 768]
        
        # 2. Image Features
        image_outputs = self.vit(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output # [batch, 768]
        
        # 3. Fusion
        combined_features = torch.cat((text_features, image_features), dim=1)
        
        # 4. Classification
        logits = self.classifier(combined_features)
        return logits