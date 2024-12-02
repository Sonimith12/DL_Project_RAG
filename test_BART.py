from transformers import BartForConditionalGeneration, BartTokenizer
from peft import PeftModel

# Load the tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

# Load the fine-tuned model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
model = PeftModel.from_pretrained(model, "./results_lora/checkpoint-120621")

# Set the model to evaluation mode
model.eval()

# Example input text
input_text = "Can you summarize the review for Cialis?"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

print("input_ids shape:", inputs["input_ids"].shape)
print("input_ids type:", type(inputs["input_ids"]))


# Generate predictions
outputs = model.generate(
    input_ids=inputs["input_ids"],  # Explicitly use keyword arguments
    max_length=250,
    num_beams=5,
    early_stopping=True
)


# Decode and print the output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model Output:", decoded_output)
