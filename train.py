import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

# import random
# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from nltk_utils import bag_of_words, tokenize, stem
# from model import NeuralNet

# # Define a function to load intents from a JSON file
# def load_intents(filename):
#     with open(filename, 'r') as f:
#         intents = json.load(f)
#     return intents

# # Define a function to preprocess intents and create training data
# def preprocess_intents(intents):
#     all_words = []
#     tags = []
#     xy = []
#     # loop through each sentence in our intents patterns
#     for intent in intents['intents']:
#         tag = intent['tag']
#         # add to tag list
#         tags.append(tag)
#         for pattern in intent['patterns']:
#             # tokenize each word in the sentence
#             w = tokenize(pattern)
#             # add to our words list
#             all_words.extend(w)
#             # add to xy pair
#             xy.append((w, tag))

#     # stem and lower each word
#     ignore_words = ['?', '.', '!']
#     all_words = [stem(w) for w in all_words if w not in ignore_words]
#     # remove duplicates and sort
#     all_words = sorted(set(all_words))
#     tags = sorted(set(tags))

#     print(len(xy), "patterns")
#     print(len(tags), "tags:", tags)
#     print(len(all_words), "unique stemmed words:", all_words)

#     # create training data
#     X_train = []
#     y_train = []
#     for (pattern_sentence, tag) in xy:
#         # X: bag of words for each pattern_sentence
#         bag = bag_of_words(pattern_sentence, all_words)
#         X_train.append(bag)
#         # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
#         label = tags.index(tag)
#         y_train.append(label)

#     X_train = torch.tensor(X_train, dtype=torch.float)
#     y_train = torch.tensor(y_train, dtype=torch.long)

#     input_size = len(X_train[0])
#     output_size = len(tags)

#     return {
#         'X_train': X_train,
#         'y_train': y_train,
#         'input_size': input_size,
#         'output_size': output_size,
#         'all_words': all_words,
#         'tags': tags
#     }

# # Define a function to train a model
# def train_model(X_train, y_train, input_size, output_size):
#     # Hyper-parameters 
#     num_epochs = 1000
#     batch_size = 8
#     learning_rate = 0.001
#     hidden_size = 8

#     class ChatDataset(Dataset):

#         def __init__(self, X_data, y_data):
#             self.n_samples = len(X_data)
#             self.x_data = X_data
#             self.y_data = y_data

#         # support indexing such that dataset[i] can be used to get i-th sample
#         def __getitem__(self, index):
#             return self.x_data[index], self.y_data[index]

#         # we can call len(dataset) to return the size
#         def __len__(self):
#             return self.n_samples

#     dataset = ChatDataset(X_train, y_train)
#     train_loader = DataLoader(dataset=dataset,
#                               batch_size=batch_size,
#                               shuffle=True)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = NeuralNet(input_size, hidden_size, output_size).to(device)

#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     # Train the model
#     for epoch in range(num_epochs):
#         for (words, labels) in train_loader:
#             words = words.to(device)
#             labels = labels.to(device)
            
#             # Forward pass
#             outputs = model(words)
#             loss = criterion(outputs, labels)
            
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        
#         if (epoch+1) % 100 == 0:
#             print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

#     print(f'final loss: {loss.item():.4f}')

#     return model

# if __name__ == "__main__":
#     # Load and preprocess intents from each dataset
#     intents_files = ['intents1.json', 'intents2.json', 'intents3.json']
#     datasets = []
#     for filename in intents_files:
#         intents = load_intents(filename)
#         preprocessed_data = preprocess_intents(intents)
#         datasets.append(preprocessed_data)

#     # Train a model for each dataset
#     trained_models = []
#     for data in datasets:
#         X_train = data['X_train']
#         y_train = data['y_train']
#         input_size = data['input_size']
#         output_size = data['output_size']
#         model = train_model(X_train, y_train, input_size, output_size)
#         trained_models.append(model)

#     # Save trained models
#     for i, model in enumerate(trained_models):
#         torch.save(model.state_dict(), f"model_{i}.pth")

#     print("Training complete. Models saved.")
