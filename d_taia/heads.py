import torch.nn as nn
import torch.nn.functional as F


class ActivityHead(nn.Module):
    def __init__(self, input_dim, num_activities, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(input_dim, num_activities)

    def forward(self, h):
        return self.head(self.dropout(h))


class TimeHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        gated = F.silu(self.W1(h)) * self.W2(h)
        h_ffn = self.norm(self.W3(gated))
        h_ffn = self.dropout(h_ffn)
        return self.out(h_ffn)