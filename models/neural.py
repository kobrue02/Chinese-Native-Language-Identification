"""LSTM and TextCNN models for NLI classification."""

import torch
import torch.nn as nn

import config

CFG = config.NEURAL_CONFIG


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with max-pool for text classification."""

    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, CFG["embed_dim"], padding_idx=0)
        self.lstm = nn.LSTM(
            CFG["embed_dim"],
            CFG["hidden_dim"],
            num_layers=CFG["num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=CFG["dropout"] if CFG["num_layers"] > 1 else 0,
        )
        self.dropout = nn.Dropout(CFG["dropout"])
        self.fc = nn.Linear(CFG["hidden_dim"] * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.dropout(self.embedding(x))  # (batch, seq, embed)
        output, _ = self.lstm(emb)  # (batch, seq, hidden*2)
        pooled, _ = output.max(dim=1)  # (batch, hidden*2)
        return self.fc(self.dropout(pooled))  # (batch, num_classes)


class TextCNN(nn.Module):
    """Multi-kernel TextCNN for text classification."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        num_filters: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, CFG["embed_dim"], padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(CFG["embed_dim"], num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(CFG["dropout"])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))  # (batch, seq, embed)
        emb = emb.transpose(1, 2)  # (batch, embed, seq)
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(emb))  # (batch, filters, seq-k+1)
            c, _ = c.max(dim=2)  # (batch, filters)
            conv_outs.append(c)
        cat = torch.cat(conv_outs, dim=1)  # (batch, filters*n_kernels)
        return self.fc(self.dropout(cat))  # (batch, num_classes)
