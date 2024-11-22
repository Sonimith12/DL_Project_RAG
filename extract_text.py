import re
from PyPDF2 import PdfReader
from collections import Counter
import json

# Define the regex pattern for headers and footers to filter out
header_footer_pattern = re.compile(r"(Page \d+|Document Title|(IMT Mines Al` es)|Footer Text)")

# Load the PDF 
# reader = PdfReader("D:/IMT MInes Ales/S9/DeepLearning/project_RAG/M2___IMT___Deep_Learning.pdf")
reader = PdfReader("D:/IMT MInes Ales/S9/Advanced Machine Learning/Deep Learning server-side inference - Jupyter Notebook.pdf")

# Store extracted text page by page
page_texts = []
for page in reader.pages:
    page_text = page.extract_text()
    # Remove lines that match the header/footer pattern
    cleaned_text = "\n".join(
        line for line in page_text.splitlines() if not header_footer_pattern.search(line)
    )
    page_texts.append(cleaned_text.splitlines())  # Split into lines for further analysis

# Analyze headers and footers
top_lines = []  # Collect the first line of each page
bottom_lines = []  # Collect the last line of each page

for page in page_texts:
    if page:  # Ensure the page is not empty
        top_lines.append(page[0])  # First line (header)
        bottom_lines.append(page[-1])  # Last line (footer)

# Count occurrences of lines
top_counts = Counter(top_lines)
bottom_counts = Counter(bottom_lines)

# Define thresholds (e.g., lines appearing on more than 2 pages)
repeated_top_lines = {line for line, count in top_counts.items() if count > 2}
repeated_bottom_lines = {line for line, count in bottom_counts.items() if count > 2}

# Prepare structured data for JSON format
structured_data = []
text = ""
for page in page_texts:
    if not page:
        continue  # Skip empty pages
    
    # Determine title (assume it's the first line not marked as a repeated header)
    title = page[1] 

    # Filter content by removing headers and footers
    filtered_content = [
        line for i, line in enumerate(page)
        if (i != 0 or line not in repeated_top_lines)  # Exclude repeated headers 
        and i != 1
        and (i != len(page) - 1 or line not in repeated_bottom_lines)  # Exclude repeated footers
    ]
    content = " ".join(filtered_content)
    # content = filtered_content
    # Append to structured data
    structured_data.append({
        "title": title.strip(),
        "content": content.strip()
    })
    text += content.strip() + "\n"
# Write to a JSON file
with open("DL_Project_RAG/Deep Learning server-side inference - Jupyter Notebook.pdf.json", "w", encoding='utf-8') as f:
    json.dump(structured_data, f, ensure_ascii=False, indent=4)

with open("DL_Project_RAG/Deep Learning server-side inference - Jupyter Notebook.pdf.txt", "w", encoding='utf-8') as f:
    f.write(text)