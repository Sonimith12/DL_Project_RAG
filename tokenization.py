from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
import json
import torch

# Load tokenizer for BART
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set PAD token if not already set

# Load data
with open("dataset/drug_data_for_rag_simplified.json", encoding="utf-8") as f:
    data = json.load(f)

# Prepare prompts (consider the target as the same text for training purpose)
prompts = [
    f"Title: {item['title']}\nContent: {item['content']}\nReview: {item['review']}\nEffectiveness Rating: {item['rating']}"
    for item in data
]

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, tokenizer, prompts, max_length):
        self.encodings = tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        item = {
            key: tensor[idx] for key, tensor in self.encodings.items()
        }
        # For sequence-to-sequence tasks, we need to use the input as labels as well
        item['labels'] = item['input_ids'].clone()  # Using input_ids as labels (autoencoder style)
        return item

# Create dataset
max_length = 512
train_dataset = TextDataset(tokenizer, prompts, max_length)

# Data collator for padding (using Seq2Seq since BART is a sequence-to-sequence model)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=None  # We don't need to specify the model for the collator
)

# Load BART model and apply LoRA (use AutoModelForSeq2SeqLM for sequence-to-sequence tasks)
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")  # Force CPU usage
lora_config = LoraConfig(
    task_type="SEQ2SEQ_LM", r=16, lora_alpha=32, lora_dropout=0.1
)
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_bart",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="no",  # No evaluation during training
    logging_dir="./logs",
    learning_rate=5e-5,
 # Disable CUDA, force CPU
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()

# Save the model with LoRA adapters
model.save_pretrained("./lora_bart")
print("Training complete. Model saved to './lora_bart'.")
