import torch
from transformers import BartForConditionalGeneration
from transformers import TrainingArguments
from transformers import Trainer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  # Specify the task type
    r=8,  # Low-rank dimension
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers (query/key/value projections)
    lora_dropout=0.1,  # Dropout for LoRA layers
    bias="none"  # Bias setup
)
model = get_peft_model(model, lora_config)
model = model.to(device)

# Move the model to the device
model = model.to(device)
def get_answer(question):
    # Tokenize the question
    inputs = tokenizer(question, return_tensors="pt", max_length=128, truncation=True)

    # Move input tensors to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Generate the response using the model
    output_ids = model.base_model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0)

    # Decode the response
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer


# Example interaction

user_question = "Hello, how are you?"
print(f"Answer: {get_answer(user_question)}")
