import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split

# Just change this file name
INPUT_JSONL = "LTX_data_2.jsonl"

# Load data from JSONL
print("Loading instruction dataset...")
data = []
"""with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        prompt = f"### Input:\n{entry['input']}\n\n### Response:\n{entry['output']}"
        data.append({"text": prompt})"""
data = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    entries = json.load(f)
    for entry in entries:
        prompt = f"### Input:\n{entry['input']}\n\n### Response:\n{entry['output']}"
        data.append({"text": prompt})

# Train/validation split
train_data, valid_data = train_test_split(data, test_size=0.1)

# Load tokenizer and model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenization
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = Dataset.from_list(train_data).map(tokenize_function, batched=True)
valid_dataset = Dataset.from_list(valid_data).map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_ate_tinyllama",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    report_to="tensorboard",
    gradient_accumulation_steps=8,
    fp16=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()
