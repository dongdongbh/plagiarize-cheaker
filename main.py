import fitz  # PyMuPDF
import os
import difflib

def compare_texts(text1, text2):
    return difflib.SequenceMatcher(None, text1, text2).ratio()

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Extract text from all PDFs in the ./data/ directory
texts = {}  # Dictionary to store filename and its extracted text
data_dir = './data/'

for filename in os.listdir(data_dir):
    if filename.endswith('.pdf'):
        file_path = os.path.join(data_dir, filename)
        texts[filename] = pdf_to_text(file_path)

# Now, compare the texts of each pair of documents
for filename1, text1 in texts.items():
    for filename2, text2 in texts.items():
        if filename1 < filename2:  # This ensures we don't compare the same pair twice or a document with itself
            similarity = compare_texts(text1, text2)
            print(f"Similarity between {filename1} and {filename2}: {similarity:.2%}")
