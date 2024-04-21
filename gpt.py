import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

bot_name = "Chetana"
print("Let's chat! (type 'quit' to exit)")
while True:
    # Get user input
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    # Tokenize input and convert to tensor
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

    # Generate response using GPT-2 model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode and print response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"{bot_name}: {response}")
