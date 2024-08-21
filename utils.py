import torch
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
import torch

def set_seed(seed: int):
    """设置所有相关库的随机种子以确保结果的可重复性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def collate_fn(batch):
    """填充短序列，补齐到本批次最长序列长度"""
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, labels



