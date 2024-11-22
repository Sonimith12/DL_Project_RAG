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
    
    content = f"Use Case: {use_case}"

    rag_data.append({
        "title": drug_name,
        "content": content,
        "review": review,
        "rating": rating,
        "approval_date": approval_date,
        "times_prescribed": times_prescribed
    })

with open("dataset/drug_data_for_rag_simplified.json", "w", encoding="utf-8") as f:
    json.dump(rag_data, f, ensure_ascii=False, indent=4)
