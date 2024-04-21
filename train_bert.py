import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Load intents data
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Extract patterns and tags
sentences = []
labels = []
label_to_index = {}
for intent in intents['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        label = intent['tag']
        if label not in label_to_index:
            label_to_index[label] = len(label_to_index)
        labels.append(label_to_index[label])

# Split data into train and test sets
train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_index))

# Tokenize input sentences
train_encodings = tokenizer(train_sentences, truncation=True, padding=True)
test_encodings = tokenizer(test_sentences, truncation=True, padding=True)

# Define custom dataset class
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create DataLoader for train and test sets
train_dataset = IntentDataset(train_encodings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = IntentDataset(test_encodings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    # Evaluate after each epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {total_loss/len(train_loader):.4f}, Test Accuracy: {accuracy:.2f}%')

# Save the trained model
model.save_pretrained("bert_intent_model")
