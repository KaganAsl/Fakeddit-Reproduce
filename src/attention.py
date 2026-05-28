import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, vit_dim=768, bert_dim=768, joint_dim=768, num_heads=8, dropout=0.1):
        super().__init__()

        self.query_proj_img = nn.Linear(vit_dim, joint_dim)
        self.key_proj_img = nn.Linear(bert_dim, joint_dim)
        self.value_proj_img = nn.Linear(bert_dim, joint_dim)

        self.query_proj_txt = nn.Linear(bert_dim, joint_dim)
        self.key_proj_txt = nn.Linear(vit_dim, joint_dim)
        self.value_proj_txt = nn.Linear(vit_dim, joint_dim)

        self.attn_img = nn.MultiheadAttention(joint_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_txt = nn.MultiheadAttention(joint_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_img = nn.LayerNorm(joint_dim)
        self.norm_txt = nn.LayerNorm(joint_dim)

        self.ffn_img = nn.Sequential(
            nn.Linear(joint_dim, joint_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(joint_dim * 4, joint_dim),
            nn.Dropout(dropout),
        )
        self.ffn_txt = nn.Sequential(
            nn.Linear(joint_dim, joint_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(joint_dim * 4, joint_dim),
            nn.Dropout(dropout),
        )

        self.norm_ffn_img = nn.LayerNorm(joint_dim)
        self.norm_ffn_txt = nn.LayerNorm(joint_dim)

    def forward(self, img_emb, txt_emb):
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1)
        if txt_emb.dim() == 2:
            txt_emb = txt_emb.unsqueeze(1)

        q_img = self.query_proj_img(img_emb)
        k_img = self.key_proj_img(txt_emb)
        v_img = self.value_proj_img(txt_emb)
        attended_img, _ = self.attn_img(q_img, k_img, v_img)
        attended_img = self.norm_img(q_img + attended_img)
        attended_img = self.norm_ffn_img(attended_img + self.ffn_img(attended_img))

        q_txt = self.query_proj_txt(txt_emb)
        k_txt = self.key_proj_txt(img_emb)
        v_txt = self.value_proj_txt(img_emb)
        attended_txt, _ = self.attn_txt(q_txt, k_txt, v_txt)
        attended_txt = self.norm_txt(q_txt + attended_txt)
        attended_txt = self.norm_ffn_txt(attended_txt + self.ffn_txt(attended_txt))

        return attended_img.squeeze(1), attended_txt.squeeze(1)