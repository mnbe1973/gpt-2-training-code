from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from datasets import load_dataset
import pandas as pd

def load_dataset():
    # Load dataset
    dataset = load_dataset('sahil2801/code_instructions_120k')
    
    # Convert a part of the dataset to a pandas DataFrame
    df = pd.DataFrame(dataset['train'])
    
    # Save this dataframe to a .csv file
    df.to_csv('code120_train.csv', index=False)
    filename = 'code120_train.csv'
    df = pd.read_csv(filename)

    # Tokenize the dataset.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenized_dataset = tokenizer(df['code'].tolist(), padding=True, truncation=True, return_tensors="pt")
    
    return tokenized_dataset

dataload=load_dataset()
    
  