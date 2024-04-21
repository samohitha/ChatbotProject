# custom_dataset.py

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

class CustomDataset(Dataset):
    def __init__(self, patterns, tokenizer, max_length=32):
        self.patterns = patterns
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        encoding = self.tokenizer(pattern, max_length=self.max_length, padding="max_length", truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # Labels are the same as input for language modeling
        }
