import torch
import torch.nn as nn


class OceanModel(nn.Module):
    def __init__(self, encoder, lexical_size):
        super().__init__()
        if lexical_size is None:
            raise ValueError("lexical_size tidak boleh None")
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.fc = nn.Linear(hidden + lexical_size, 5)

    def forward(self, input_ids, attention_mask, lexical):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        x = torch.cat([cls, lexical], dim=1)
        return self.fc(x)
