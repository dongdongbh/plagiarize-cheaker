import argparse

import fitz  # PyMuPDF
import csv
import os
import spacy
import shutil
from shutil import copy2
from tqdm import tqdm
import difflib
from multiprocessing import Pool, cpu_count
# import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


# Load a spaCy language model
nlp = spacy.load("en_core_web_sm")  # or "en_core_web_lg" for better accuracy but requires more memory


question_path = './data/questions/Homework6.pdf'
data_dir = './data/hw6/'
cover_letter_path = './data/questions/675cover.pdf'


def get_args():
    parser = argparse.ArgumentParser(description="Plagiarism Detection Tool")
    parser.add_argument("--question_path", type=str, default='./data/questions/Homework6.pdf', help="Path to the PDF file containing the homework questions.")
    parser.add_argument("--data_dir", type=str, default='./data/hw6/', help="Directory containing student submissions.")
    parser.add_argument("--cover_letter_path", type=str, default='./data/questions/675cover.pdf', help="Path to the PDF file containing the cover letter.")

    parser.add_argument("--min_block_size", type=int, default=50, help="Minimum block size for considering a text match significant")
    parser.add_argument("--question_filter_threshold", type=float, default=0.7, help="Threshold for filtering out question sentences")
    parser.add_argument("--similarity_threshold", type=float, default=0.2, help="Threshold for considering documents similar")
    # Add other arguments as necessary
    args = parser.parse_args()
    return args

def remove_review_folders(data_dir):
    review_folders_path = os.path.join(data_dir, "review_folders")
    try:
        # Remove the directory and all its contents
        shutil.rmtree(review_folders_path)
        print(f"Removed existing directory: {review_folders_path}")
    except FileNotFoundError:
        # Directory does not exist, no action needed
        print(f"No existing directory to remove: {review_folders_path}")





def append_to_global_csv(global_csv_path, data):
    with open(global_csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def handle_individual_folders(data_dir, student_id, assignment_number, peer_id, similarity_score, matched_sections, submission_path, peer_submission_path):
    """
    Handles the creation of individual folders for plagiarism review, including copying relevant files
    and updating a local CSV report.

    Args:
        data_dir (str): Base directory for data processing.
        student_id (str): Identifier for the student.
        assignment_number (str or int): The assignment number.
        peer_id (str): Identifier for the peer student for comparison.
        similarity_score (float): The calculated similarity score between the submissions.
        matched_sections ([str]): List of text sections that matched.
        submission_path (str): File path to the student's submission.
        peer_submission_path (str): File path to the peer's submission.
    """
    # Create a specific folder for the student under review
    student_folder = os.path.join(data_dir, "review_folders", student_id)
    os.makedirs(student_folder, exist_ok=True)

    # Copy the student's and the peer's submissions into this folder
    copy2(submission_path, os.path.join(student_folder, os.path.basename(submission_path)))
    copy2(peer_submission_path, os.path.join(student_folder, os.path.basename(peer_submission_path)))

    # Maintain a local CSV file within the student's folder for detailed reporting
    local_csv_path = os.path.join(student_folder, 'report.csv')
    with open(local_csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the CSV is newly created
        if os.stat(local_csv_path).st_size == 0:
            writer.writerow(['Student ID', 'Peer ID', 'Assignment Number', 'Similarity Score', 'Matched Sections'])
        # Append details of the current comparison
        writer.writerow([student_id, peer_id, assignment_number, similarity_score, "; ".join(matched_sections)])


def get_sentences(pdf_path):
    question_text = pdf_to_text(pdf_path)
    sentences = sent_tokenize(question_text)
    return sentences


def filter_sentences(text, junk_sentences, question_filter_threshold):
    """
    Filters out sentences from 'text' that are similar to 'junk_sentences'
    based on a given similarity threshold.

    Args:
    - text (str): The text to filter.
    - junk_sentences (list of str): Sentences to be considered as 'junk'.
    - question_filter_threshold (float): The threshold for filtering sentences.

    Returns:
    - str: The filtered text.
    """
    filtered_sentences = []
    doc = nlp(text)
    for sent in doc.sents:
        if not any(difflib.SequenceMatcher(None, str(sent), junk).ratio() > question_filter_threshold for junk in junk_sentences):
            filtered_sentences.append(str(sent))
    return " ".join(filtered_sentences)

def is_cover_letter(filename):
    # Determines if the file is a cover letter based on the filename
    return 'cover' in filename.lower()

# Original pdf_to_text function for general text extraction
def pdf_to_text(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def remove_sentences(submission_text, question_sentences):
    submission_sentences = sent_tokenize(submission_text)
    filtered_sentences = [s for s in submission_sentences if s not in question_sentences]
    return ' '.join(filtered_sentences)

def compare_texts(data):
    user1, text1, user2, text2, similarity_threshold, min_block_size = data
    matcher = difflib.SequenceMatcher(None, text1, text2)
    similarity = matcher.ratio()

    if similarity > similarity_threshold:
        matched_sections = []
        for block in matcher.get_matching_blocks():
            if block.size > min_block_size:  # Use the dynamically provided MIN_BLOCK_SIZE
                start1, start2, size = block
                matched_text1 = text1[start1:start1+size]
                matched_sections.append(f"Start: {start1}, Match: {matched_text1}")
        return (user1, user2, similarity, matched_sections)
    else:
        return None



def generate_comparison_pairs(texts):
    # Generate all unique pairs of texts for comparison
    keys = list(texts.keys())
    print(f"Total number of files to compare: {len(keys)}")
    pairs = [(keys[i], texts[keys[i]], keys[j], texts[keys[j]]) for i in range(len(keys)) for j in range(i+1, len(keys))]
    return pairs

def process_and_filter_text(args):
    """
    Function to first extract text from a PDF and then filter sentences using a threshold.
    This is designed to be used with multiprocessing.
    
    Args:
        args (tuple): A tuple containing the filename, the directory of the file, 
                      the list of sentences to filter out, and the similarity threshold.
                      
    Returns:
        tuple: The filename and its filtered text, or None if text extraction fails.
    """
    # Adjust the unpacking to include the threshold
    filename, data_dir, question_sentences, question_filter_threshold = args
    pdf_path = os.path.join(data_dir, filename)
    text = pdf_to_text(pdf_path)
    
    if text is not None:
        # Now pass the threshold to filter_sentences
        filtered_text = filter_sentences(text, question_sentences, question_filter_threshold)
        return filename, filtered_text
    else:
        return filename, None

def process_pdfs(data_dir, question_sentences, question_filter_threshold):
    files_to_process = [(filename, data_dir, question_sentences, question_filter_threshold) for filename in os.listdir(data_dir) if filename.endswith('.pdf') and not is_cover_letter(filename)]

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_and_filter_text, files_to_process), total=len(files_to_process)))
    
    # Initialize dictionaries for the results
    submission_paths = {}
    filtered_texts = {}
    
    for filename, filtered_text in results:
        if filtered_text is not None:
            username = filename.split('_')[1]
            submission_paths[username] = os.path.join(data_dir, filename)
            filtered_texts[username] = filtered_text
    
    return filtered_texts, submission_paths

def main():

    args = get_args()
    global_csv_path = os.path.join(args.data_dir, 'plagiarism_report.csv')

    with open(global_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Student ID 1', 'Student ID 2', 'Assignment Number', 'Similarity Score', 'Matched Sections', 'Explanatory Note'])

    remove_review_folders(args.data_dir)

    question_sentences = get_sentences(args.question_path)

    cover_letter_sentences = get_sentences(args.cover_letter_path)

    combined_filter_sentences = question_sentences + cover_letter_sentences

    print("Preparing data...")

    # Assuming texts is a dictionary where the key is the username and the value is the text
    texts, submission_paths = process_pdfs(args.data_dir, combined_filter_sentences, args.question_filter_threshold)
    
    # Generate comparison pairs
    pairs = generate_comparison_pairs(texts)
    print(f"Total comparisons to perform: {len(pairs)}")

    print("Comparing data...")
    
    comparison_data = [(user1, text1, user2, text2, args.similarity_threshold, args.min_block_size) for user1, text1, user2, text2 in pairs]

    
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(compare_texts, comparison_data), total=len(comparison_data)))
    # Filter out None results (comparisons with similarity <= 25%)
    filtered_results = [result for result in results if result is not None]

    # Processing the filtered results
    for user1, user2, similarity, matched_sections in filtered_results:
        assignment_number = data_dir  # Assign the actual assignment number
        data = [user1, user2, assignment_number, similarity, "; ".join(matched_sections), "Potential plagiarism detected."]
        append_to_global_csv(global_csv_path, data)
        handle_individual_folders(args.data_dir, user1, assignment_number, user2, similarity, matched_sections, submission_paths[user1], submission_paths[user2])
        handle_individual_folders(args.data_dir, user2, assignment_number, user1, similarity, matched_sections, submission_paths[user2], submission_paths[user1])
    
       
    # Print or process your results
    for result in filtered_results:
        user1, user2, similarity, _ = result
        print(f"Similarity between {user1} and {user2}: {similarity:.2%}")

    # Count the number of comparisons with similarity > 25%
    count_of_plagiarize = len(filtered_results)

    # Print the count
    print(f"Total document pairs with more than {args.similarity_threshold} similarity: {count_of_plagiarize}")


if __name__ == "__main__":
    main()
