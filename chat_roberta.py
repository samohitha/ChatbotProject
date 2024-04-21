# # import torch
# # from transformers import RobertaTokenizer, RobertaForSequenceClassification
# # import json
# # import random

# # # Load intents dataset
# # with open('intents.json', 'r') as json_data:
# #     intents = json.load(json_data)

# # # Load trained RoBERTa model
# # model_path = 'roberta_intent_model.pth'
# # model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(intents['intents']))
# # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# # model.eval()

# # # Load RoBERTa tokenizer
# # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# # # Chat with the user
# # bot_name = "Chetana"
# # print("Let's chat! (type 'quit' to exit)")
# # while True:
# #     user_input = input("You: ")
# #     if user_input == "quit":
# #         break

# #     # Tokenize input
# #     tokenized_input = tokenizer.encode(user_input, add_special_tokens=True)

# #     # Predict intent
# #     with torch.no_grad():
# #         output = model(torch.tensor([tokenized_input]))[0]
# #     predicted_class = torch.argmax(output, dim=1).item()
# #     intent_tag = intents['intents'][predicted_class]['tag']

# #     # Get a random response for the predicted intent
# #     responses = [response for intent in intents['intents'] if intent['tag'] == intent_tag for response in intent['responses']]
# #     response = random.choice(responses)
# #     print(f"{bot_name}: {response}")


# import torch
# from transformers import RobertaTokenizer, RobertaForSequenceClassification
# import json
# import random

# # Load intents dataset
# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# # Load trained RoBERTa model
# model_path = 'roberta_intent_model.pth'
# model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(intents['intents']))
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()

# # Load RoBERTa tokenizer
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base', padding_side='left')

# # Chat with the user
# bot_name = "Chetana"
# print("Let's chat! (type 'quit' to exit)")
# while True:
#     user_input = input("You: ")
#     if user_input == "quit":
#         break

#     # Tokenize input
#     tokenized_input = tokenizer.encode(user_input, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)

#     # Predict intent
#     with torch.no_grad():
#         output = model(torch.tensor([tokenized_input]))[0]
#     predicted_class = torch.argmax(output, dim=1).item()
#     intent_tag = intents['intents'][predicted_class]['tag']

#     # Get a random response for the predicted intent
#     intent = next((intent for intent in intents['intents'] if intent['tag'] == intent_tag), None)
#     if intent:
#         response = random.choice(intent['responses'])
#     else:
#         response = "Sorry, I didn't understand that."
#     print(f"{bot_name}: {response}")


# ===========================code2===================

# import torch
# import json
# import random
# from transformers import RobertaTokenizer, RobertaForSequenceClassification

# # Load intents dataset
# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# # Load trained RoBERTa model
# model_path = 'roberta_intent_model.pth'
# model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(intents['intents']))
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# model.eval()

# # Load RoBERTa tokenizer
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# # Map numeric IDs to intent tags
# id_to_tag = {idx: intent['tag'] for idx, intent in enumerate(intents['intents'])}

# # Chat with the user
# bot_name = "Chetana"
# print("Let's chat! (type 'quit' to exit)")
# while True:
#     user_input = input("You: ")
#     if user_input == "quit":
#         break

#     # Tokenize input
#     tokenized_input = tokenizer.encode(user_input, add_special_tokens=True)

#     # Predict intent
#     with torch.no_grad():
#         output = model(torch.tensor([tokenized_input]))[0]
#     predicted_class = torch.argmax(output, dim=1).item()
#     intent_tag = id_to_tag[predicted_class]

#     # Get a random response for the predicted intent
#     responses = [response for intent in intents['intents'] if intent['tag'] == intent_tag for response in intent['responses']]
#     response = random.choice(responses)
#     print(f"{bot_name}: {response}")


# ===========================code 3 ======================================================

import torch
import json
import random
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load intents dataset
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load trained RoBERTa model
model_path = 'roberta_intent_model.pth'
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(intents['intents']))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Map numeric IDs to intent tags
id_to_tag = {idx: intent['tag'] for idx, intent in enumerate(intents['intents'])}

# Chat with the user
bot_name = "Chetana"
print("Let's chat! (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input == "quit":
        break

    # Tokenize input
    tokenized_input = tokenizer.encode(user_input, add_special_tokens=True)

    # Predict intent
    with torch.no_grad():
        output = model(torch.tensor([tokenized_input]))[0]
    predicted_class = torch.argmax(output, dim=1).item()
    intent_tag = id_to_tag[predicted_class]

    # Get a random response for the predicted intent
    responses = [intent['responses'] for intent in intents['intents'] if intent['tag'] == intent_tag]
    response_list = [response for sublist in responses for response in sublist]  # Flatten responses
    response = random.choice(response_list)
    print(f"{bot_name}: {response}")
