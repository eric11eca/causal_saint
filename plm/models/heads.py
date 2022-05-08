from __future__ import annotations

import abc
import torch
import torch.nn as nn


class BaseHead(nn.Module, metaclass=abc.ABCMeta):
    """Absract class for task heads"""

    @abc.abstractmethod
    def __init__(self):
        super().__init__()


class ClassificationHead(BaseHead):
    def __init__(self, task, hidden_size, hidden_dropout_prob, **kwargs):
        """From RobertaClassificationHead"""
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, len(task.LABELS))
        self.num_labels = len(task.LABELS)

    def forward(self, pooled):
        x = self.dropout(pooled)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


class BaseMLMHead(BaseHead, metaclass=abc.ABCMeta):
    pass
