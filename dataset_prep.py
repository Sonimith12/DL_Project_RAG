import pandas as pd
import json

data = pd.read_csv("dataset/train.csv", sep=",")

rag_data = []
for _, row in data.iterrows():
    drug_name = row["name_of_drug"]
    use_case = row["use_case_for_drug"]
    review = row["review_by_patient"]
    rating = row["effectiveness_rating"]
    approval_date = row["drug_approved_by_UIC"]
    times_prescribed = row["number_of_times_prescribed"]
    
    content = f"Use Case: {use_case}\nReview: {review}\nEffectiveness Rating: {rating}/10\nApproved By UIC: {approval_date}\nTimes Prescribed: {times_prescribed}"

    rag_data.append({
        "title": drug_name,
        "content": content
    })

with open("dataset/drug_data_for_rag.json", "w", encoding="utf-8") as f:
    json.dump(rag_data, f, ensure_ascii=False, indent=4)
