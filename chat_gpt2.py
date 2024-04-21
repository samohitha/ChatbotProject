import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

bot_name = "Chetana"
print("Let's chat! (type 'quit' to exit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
    # Generate response using the GPT-2 model
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, 
                            early_stopping=True)

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"{bot_name}: {response}")
