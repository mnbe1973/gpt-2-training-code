import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from torch.utils.data import Dataset
import torch

class GPT2Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# Loading and tokenizing data
filename = 'code120_train.csv'
df = pd.read_csv(filename)  
#df['output'] = df['output'].astype(str)
#output=df['output'].tolist()
#instruction=df['instruction'].tolist()
#df = df[df['output'].notna()]
df['input_output'] = df['instruction'].astype(str) + " </s> " + df['output'].astype(str)  # Join the instruction and code together

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
tokenizer.pad_token = tokenizer.eos_token
encodings = tokenizer(df['input_output'].tolist(), padding=True, truncation=True, return_tensors="pt")

# Creating Dataset
dataset = GPT2Dataset(encodings)  # Change here

# Define the model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
)

# Define the trainer
trainer = Trainer(
    model=model,                          # the instantiated Transformers model to be trained
    args=training_args,                   # training arguments, defined above
    train_dataset=dataset,                # training dataset
)

# Train the model
trainer.train()
output_dir = './saved_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)