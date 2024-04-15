import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Chetana"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")

# import random
# import json
# import torch
# from model import NeuralNet
# from nltk_utils import bag_of_words, tokenize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load intents from multiple JSON files
# intents_files = ['intents1.json', 'intents2.json', 'intents3.json']
# all_intents = []
# for file in intents_files:
#     with open(file, 'r') as json_data:
#         intents_data = json.load(json_data)
#         all_intents.extend(intents_data['intents'])

# # Preprocess intents from all files
# all_words = []
# tags = []
# patterns_map = {}
# for intent_data in all_intents:
#     tag = intent_data['tag']
#     tags.append(tag)
#     patterns = intent_data['patterns']
#     for pattern in patterns:
#         words = tokenize(pattern)
#         all_words.extend(words)
#         if tag not in patterns_map:
#             patterns_map[tag] = []
#         patterns_map[tag].append(pattern)  # Modified to keep original patterns

# # Remove duplicates and sort words
# all_words = sorted(set(all_words))

# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# model_state = data["model_state"]

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "Chetana"
# print("Let's chat! (type 'quit' to exit)")
# while True:
#     sentence = input("You: ")
#     if sentence == "quit":
#         break

#     sentence_words = tokenize(sentence)
#     X = bag_of_words(sentence_words, all_words)
    
#     # Ensure the X vector has the same length as the input_size expected by the model
#     # If not, pad or truncate the vector to match input_size
#     if len(X) < input_size:
#         # Pad X with zeros to match input_size
#         X = X + [0] * (input_size - len(X))
#     elif len(X) > input_size:
#         # Truncate X to match input_size
#         X = X[:input_size]
    
#     X = torch.tensor(X, dtype=torch.float).reshape(1, -1).to(device)  # Reshape to match expected input shape
    
#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]

#     if prob.item() > 0.75 and tag in patterns_map:
#         responses = patterns_map[tag]
#         print(f"{bot_name}: {random.choice(responses)}")
#     else:
#         print(f"{bot_name}: I do not understand...")
