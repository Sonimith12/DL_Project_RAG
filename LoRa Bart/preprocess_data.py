import json
from transformers import BartTokenizer

def preprocess_data(input_json_path, output_dir):
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
    with open(input_json_path, encoding="utf-8") as f:
        data = json.load(f)

    qa_data = []
    for item in data:
        medication = item["title"]
        content = item["content"]

        # Extract structured questions and answers
        questions_and_answers = [
            {"question": f"What is the use case of {medication}?", 
             "answer": content.split("Use Case: ")[1].split("\n")[0]},
            {"question": f"What is the effectiveness rating of {medication}?", 
             "answer": content.split("Effectiveness Rating: ")[1].split("\n")[0]},
            {"question": f"Can you summarize the review for {medication}?", 
             "answer": content.split("Review: ")[1].split("Effectiveness Rating:")[0].strip()},
            {"question": f"How many times has {medication} been prescribed?", 
             "answer": content.split("Times Prescribed: ")[1].split("\n")[0]},
            {"question": f"When was {medication} approved by UIC?", 
             "answer": content.split("Approved By UIC: ")[1].split("\n")[0]}
        ]
        qa_data.extend(questions_and_answers)

    # Save the question-answer pairs as a JSON file
    qa_json_path = f"{output_dir}/questions.json"
    with open(qa_json_path, "w") as qa_file:
        json.dump(qa_data, qa_file, indent=4)

    print(f"Processed question-answer pairs saved to: {qa_json_path}")

    # Tokenize the questions and answers
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    train_encodings = tokenizer([d["question"] for d in qa_data], truncation=True, padding=True, max_length=128)
    train_labels = tokenizer([d["answer"] for d in qa_data], truncation=True, padding=True, max_length=128)

    # Save tokenized data
    torch.save({"inputs": train_encodings, "labels": train_labels}, f"{output_dir}/train_encodings.pt")
    print(f"Tokenized training data saved to: {output_dir}/train_encodings.pt")

if __name__ == "__main__":
    import argparse
    import torch
    import os

    # Argument parser
    parser = argparse.ArgumentParser(description="Preprocess JSON data for BART fine-tuning.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed files.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run preprocessing
    preprocess_data(args.input_json, args.output_dir)
