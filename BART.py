import torch
from transformers import BartForConditionalGeneration
from transformers import TrainingArguments
from transformers import Trainer
from peft import get_peft_model, LoraConfig, TaskType

import os
os.environ["WANDB_DISABLED"] = "true"

# Load tokenized data
data = torch.load("./dataset/train_encodings.pt")
inputs = data["inputs"]
labels = data["labels"]

# Define dataset class
class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels["input_ids"][idx])
        return item

# Prepare dataset
train_dataset = QADataset(inputs, labels)

# Continue with model fine-tuning...
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # Specify the task type
    r=8,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers (query/key/value projections)
    lora_dropout=0.1,  # Dropout for LoRA layers
    bias="none"  # Bias setup
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results_lora",
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs_lora",
    fp16=True,  # Use mixed precision for faster training
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model with LoRA
trainer.train()
