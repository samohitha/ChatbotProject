import random
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load intents data
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert_intent_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

bot_name = "Chetana"
print("Let's chat! (type 'quit' to exit)")

while True:
    # Get user input
    user_input = input("You: ")

    # Check if user wants to quit
    if user_input.lower() == "quit":
        break

    # Tokenize input sentence
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    inputs.to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        predicted_tag = model.config.id2label[predicted_class]
        
        print("Predicted Tag:", predicted_tag)

    # Find the corresponding intent and response
    response = "I'm sorry, I'm not sure how to respond to that."
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            response = random.choice(intent['responses'])
            break

    # Print bot response
    print(f"{bot_name}: {response}")
