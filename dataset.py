import torch
from torch.utils.data import Dataset
from collections import Counter
import pandas as pd
import re
import os
import pickle


class TextDataset(Dataset):
    def __init__(self, file_path, max_len=50, mode="test"):
        """
        :param file_path: 训练数据文件路径
        :param max_len: 句子最大长度，句子长度大于max_len的会被截断
        :param mode: 训练模式还是测试模式
        """
        
        self.load_data(file_path=file_path) # 加载数据 数据一次性加载到内存
        self.max_len = max_len 
        
        if os.path.exists('label_map.txt'):  # 保存第一次训练时的映射关系，方便推理时获取原来的标签
            with open("label_map.txt", "r", encoding="utf-8") as f:
                self.label_to_index = eval(f.readline())
        else:
            self.label_to_index = {label: idx for idx, label in enumerate(set(self.labels))}
            with open("label_map.txt", "w", encoding="utf-8") as f:
                f.write(str(self.label_to_index))
                
        if mode == "train": # 训练模式下构建词汇表
            self.build_vocab(self.texts)
        else:
            if not os.path.exists('vocab.pkl'):
                raise FileNotFoundError("No vocab file found. Please train first.")
            with open('vocab.pkl', 'rb') as f:
                self.vocab = pickle.load(f)

    def build_vocab(self, texts, min_freq=1):
        """构建词汇表"""
        counter = Counter()
        for text in texts:
            for char in text:
                counter[char] += 1
        vocab = {char: idx for idx, (char, freq) in enumerate(counter.items()) if freq >= min_freq}
        vocab['<PAD>'] = len(vocab)
        vocab['<UNK>'] = len(vocab)
        self.vocab = vocab
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)

    def load_data(self,file_path):
        data = pd.read_csv(file_path, sep='\t')  # 数据为 TSV 格式
        self.texts = data['语料'].tolist()
        self.labels = data['意图'].tolist()


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_indices = [self.vocab.get(char, self.vocab['<UNK>']) for index, char in enumerate(text) if index < self.max_len]
        return torch.tensor(text_indices), torch.tensor(self.label_to_index.get(label))


