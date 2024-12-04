import json
import torch
from transformers import BartTokenizer

def preprocess_data():
    """
    Preprocesses the input JSON file for fine-tuning a BART model on a QA task.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_dir (str): Directory to save the processed files.

    Outputs:
        - `questions.json`: Contains the processed question-answer pairs.
        - `train_encodings.pt`: Encoded questions and answers for training.
    """
    # Load the JSON dataset
    input_json_path = "C:\\Users\\HP\\Documents\\EMA cours\\Deep learning\\question_answer_pairs_truncated.json"

    with open(input_json_path, encoding="utf-8") as f:
        data = json.load(f)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    train_encodings = tokenizer([d["question"] for sublist in data for d in sublist], truncation=True, padding=True, max_length=128)
    train_labels = tokenizer([d["answer"] for sublist in data for d in sublist], truncation=True, padding=True, max_length=128)

    # Save tokenized data
    torch.save({"inputs": train_encodings, "labels": train_labels}, f"C:\\Users\\HP\\Documents\\EMA cours\\Deep learning\\train_encodings_truncated.pt")
    


preprocess_data()
