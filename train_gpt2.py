import torch
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load the dataset
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Concatenate all patterns into one text corpus
corpus = ""
for intent in intents['intents']:
    patterns = intent['patterns']
    corpus += " ".join(patterns) + " "

# Tokenize the corpus
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenized_corpus = tokenizer.encode(corpus, return_tensors="pt", max_length=1024, truncation=True)

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, tokenized_corpus, tokenizer):
        self.tokenized_corpus = tokenized_corpus
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokenized_corpus)

    def __getitem__(self, idx):
        return self.tokenized_corpus[idx]

# Create DataLoader
dataset = CustomDataset(tokenized_corpus, tokenizer)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load pre-trained GPT-2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Fine-tune the model
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

num_epochs = 3  # You can adjust this
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        input_ids = batch.to(device)
        labels = input_ids.clone()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

# Save the fine-tuned model
model_path = "fine_tuned_gpt2_model.pth"
model.save_pretrained(model_path)

