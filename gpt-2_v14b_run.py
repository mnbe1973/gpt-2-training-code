from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer and model
model_name_or_path = './saved_model'  # Replace with the path to your trained model checkpoint
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

def generate_text(prompt, max_length=100, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, temperature=temperature)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example prompt to generate text
prompt = "Create a function in C to calculate the factorial of a number."

generated_text = generate_text(prompt)
print(generated_text)