# models_rs/adni_smoke_test_multimodal_token_assembler_122625_rs.py

import torch
import torch.nn as nn

class MultimodalTokenAssembler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tokens_mri, tokens_tab):
        """
        tokens_mri: (B, 512, d_model)
        tokens_tab: (B, 10, d_model)
        """
        return torch.cat([tokens_mri, tokens_tab], dim=1)
