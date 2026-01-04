import torch
import torch.nn as nn

class TabularTokenEncoder(nn.Module):
    def __init__(self, d_model=16):
        super().__init__()

        # ===== categorical embeddings =====
        self.sex_embed = nn.Embedding(2, d_model)
        self.ptmarry_embed = nn.Embedding(4, d_model)
        self.apoe4_embed = nn.Embedding(3, d_model)

        # ===== numeric projections =====
        self.num_proj = nn.ModuleDict({
            "Age": nn.Linear(1, d_model),
            "PTEDUCAT": nn.Linear(1, d_model),
            "VSWEIGHT": nn.Linear(1, d_model),
            "VSBPSYS": nn.Linear(1, d_model),
            "VSBPDIA": nn.Linear(1, d_model),
            "VSPULSE": nn.Linear(1, d_model),
            "VSRESP": nn.Linear(1, d_model),
        })

        # ===== output normalization =====
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tab_c, tab_x):
        tokens = []

        # --- categorical tokens ---
        tokens.append(self.sex_embed(tab_c["Sex"]))
        tokens.append(self.ptmarry_embed(tab_c["PTMARRY"]))
        tokens.append(self.apoe4_embed(tab_c["APOE4"]))

        # --- numeric tokens ---
        for k, proj in self.num_proj.items():
            x = tab_x[k].unsqueeze(1)   # (B, 1)
            tokens.append(proj(x))      # (B, d_model)

        # (B, N_tab, d_model)
        tok = torch.stack(tokens, dim=1)

        # normalization for stability
        tok = self.out_norm(tok)

        return tok
