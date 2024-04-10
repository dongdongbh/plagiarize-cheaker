import argparse
from docx import Document
import fitz  # PyMuPDF
import csv
import json
import os
import spacy
import shutil
from shutil import copy2
from tqdm import tqdm
import difflib
from multiprocessing import Pool, cpu_count
# import re
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize


# Load a spaCy language model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # or "en_core_web_lg" for better accuracy but requires more memory
nlp.add_pipe('sentencizer')
# spacy.require_gpu()

# Global dictionary for detailed tracking
# Format: {student_id: [(hw_number, peer_id, similarity), ...], ...}
detailed_cheating_instances = {}


def update_detailed_cheating_instances(user1, user2, hw_number, similarity):
    if user1 not in detailed_cheating_instances:
        detailed_cheating_instances[user1] = []
    if user2 not in detailed_cheating_instances:
        detailed_cheating_instances[user2] = []
    
    detailed_cheating_instances[user1].append((hw_number, user2, similarity))
    detailed_cheating_instances[user2].append((hw_number, user1, similarity))


def generate_enhanced_global_report(data_dir):
    global_csv_path = os.path.join(data_dir, 'global-report.csv')

    print(f"Generating enhanced global report to file {global_csv_path}")
    
    with open(global_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Student ID', 'Cheating Frequency', 'Details'])

        for student_id, details in detailed_cheating_instances.items():
            # Aggregate details by homework
            hw_details = {}
            for hw_number, peer_id, similarity in details:
                if hw_number not in hw_details:
                    hw_details[hw_number] = []
                hw_details[hw_number].append((peer_id, similarity))
            
            # Compile details string
            details_str = "; ".join([
                f"HW{hw}: {', '.join([f'{peer_id} (Similarity: {similarity:.2%})' for peer_id, similarity in peers])}" 
                for hw, peers in hw_details.items()
            ])
            writer.writerow([student_id, len(hw_details), details_str])

def get_args():
    parser = argparse.ArgumentParser(description="Plagiarism Detection Tool")
    parser.add_argument("--config_file", type=str, default="./config.json", help="Path to the JSON configuration file. Default: './config.json'")
    args = parser.parse_args()
    return args

def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    return config

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
    # Create a specific folder for the student under review
    student_folder = os.path.join(data_dir, "review_folders", student_id)
    os.makedirs(student_folder, exist_ok=True)

    # Define the target paths for the symbolic links
    target_submission_path = os.path.join(student_folder, os.path.basename(submission_path))
    target_peer_submission_path = os.path.join(student_folder, os.path.basename(peer_submission_path))

    # Function to create or replace a symbolic link
    def create_or_replace_symlink(source, target):
        if os.path.islink(target):
            os.unlink(target)  # Remove the existing symbolic link
        elif os.path.exists(target):
            os.remove(target)  # Remove the existing file (use with caution)
        os.symlink(source, target)

    # Create or replace symbolic links
    create_or_replace_symlink(submission_path, target_submission_path)
    create_or_replace_symlink(peer_submission_path, target_peer_submission_path)

    # Maintain a local CSV file within the student's folder for detailed reporting
    local_csv_path = os.path.join(student_folder, 'report.csv')
    with open(local_csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write header if the CSV is newly created
        if os.stat(local_csv_path).st_size == 0:
            writer.writerow(['Student ID', 'Peer ID', 'Assignment Number', 'Similarity Score', 'Matched Sections'])
        # Append details of the current comparison
        writer.writerow([student_id, peer_id, assignment_number, similarity_score, "; ".join(matched_sections)])



def get_sentences(pdf_path, nlp):
    question_text = pdf_to_text(pdf_path)
    doc = nlp(question_text)
    sentences = [str(sent) for sent in doc.sents]
    return sentences


def filter_sentences(text, junk_sentences, question_filter_threshold):
    filtered_sentences = []
    doc = nlp(text)
    for sent in doc.sents:
        if not any(difflib.SequenceMatcher(None, str(sent), junk).ratio() > question_filter_threshold for junk in junk_sentences):
            filtered_sentences.append(str(sent))
    return " ".join(filtered_sentences)

def is_cover_letter(filename):
    # Determines if the file is a cover letter based on the filename
    return 'cover' in filename.lower()

def docx_to_text(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

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

def file_to_text(file_path):
    if file_path.endswith('.pdf'):
        return pdf_to_text(file_path)
    elif file_path.endswith('.docx'):
        return docx_to_text(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return None

def compare_texts(data):
    user1, text1, user2, text2, similarity_threshold, min_block_size = data

    # doc1 = nlp(text1)
    # doc2 = nlp(text2)
    # # Calculate semantic similarity
    # similarity = doc1.similarity(doc2)


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
    filename, data_dir, question_sentences, question_filter_threshold = args
    file_path = os.path.join(data_dir, filename)
    text = file_to_text(file_path)  # Generalized text extraction
    
    if text is not None:
        filtered_text = filter_sentences(text, question_sentences, question_filter_threshold)
        return filename, filtered_text
    else:
        return filename, None

def process_pdfs(data_dir, question_sentences, question_filter_threshold):
    files_to_process = [
        (filename, data_dir, question_sentences, question_filter_threshold) 
        for filename in os.listdir(data_dir) 
        if (filename.endswith('.pdf') or filename.endswith('.docx')) and not is_cover_letter(filename)
    ]

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

def process_single_homework(data_dir, question_path, cover_letter_path, similarity_threshold, min_block_size, question_filter_threshold, hw_number):
    global_csv_path = os.path.join(data_dir, 'plagiarism_report.csv')

    # Initialize or clear the global CSV file for this homework
    with open(global_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Student ID 1', 'Student ID 2', 'Assignment Number', 'Similarity Score', 'Matched Sections', 'Explanatory Note'])

    # You might still want to remove review folders here or ensure they're clean before starting
    remove_review_folders(data_dir)

    # Extract sentences from question and cover letter PDFs
    question_sentences = get_sentences(question_path, nlp)
    cover_letter_sentences = get_sentences(cover_letter_path, nlp)
    combined_filter_sentences = question_sentences + cover_letter_sentences
    print("Preparing data...")

    # Process PDFs/DOCXs and compare texts
    texts, submission_paths = process_pdfs(data_dir, combined_filter_sentences, question_filter_threshold)
    pairs = generate_comparison_pairs(texts)
    print(f"Total comparisons to perform: {len(pairs)}")

    print("Comparing data...")
    comparison_data = [(user1, text1, user2, text2, similarity_threshold, min_block_size) for user1, text1, user2, text2 in pairs]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(compare_texts, comparison_data), total=len(comparison_data)))

    filtered_results = [result for result in results if result is not None]

    # Process results and update the global CSV
    for user1, user2, similarity, matched_sections in filtered_results:
        assignment_number = os.path.basename(data_dir)  # Extract assignment number from directory name
        data = [user1, user2, assignment_number, similarity, "; ".join(matched_sections), "Potential plagiarism detected."]
        append_to_global_csv(global_csv_path, data)
        update_detailed_cheating_instances(user1, user2, hw_number, similarity)
        handle_individual_folders(data_dir, user1, assignment_number, user2, similarity, matched_sections, submission_paths[user1], submission_paths[user2])
        handle_individual_folders(data_dir, user2, assignment_number, user1, similarity, matched_sections, submission_paths[user2], submission_paths[user1])

    # Print summary statistics for this homework
    print_summary_statistics(filtered_results)

def print_summary_statistics(filtered_results):
    involved_students = {user for result in filtered_results for user in result[:2]}
    print(f"Total document pairs with more than specified similarity: {len(filtered_results)}")
    print(f"Total number of unique students involved: {len(involved_students)}")

def process_homework_dirs(data_dir, similarity_thresholds, min_block_size, question_filter_threshold):

    hw_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('hw')]
    cover_letter_path = os.path.join(data_dir, "questions", "675cover.pdf")
    
    for hw_dir_name in hw_dirs:
        hw_dir = os.path.join(data_dir, hw_dir_name)
        hw_number = hw_dir_name[2:]  # Extract the number from the directory name, e.g., '1' from 'hw1'
        current_assignment = f"HW{hw_number}"
        current_threshold = similarity_thresholds.get(current_assignment, 0.15)  # Use the provided threshold or default to 0.15

        question_path = os.path.join(data_dir, "questions", f"Homework{hw_number}.pdf")
        
        
        print(f"------------------Processing {hw_dir} with questions from {question_path}--------------------")
        process_single_homework(hw_dir, question_path, cover_letter_path, current_threshold, min_block_size, question_filter_threshold, hw_number)

if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_file)

    # Now you can use config dict directly to access your settings
    data_dir = config["data_dir"]
    min_block_size = config["min_block_size"]
    question_filter_threshold = config["question_filter_threshold"]
    similarity_thresholds = config["similarity_thresholds"]

    # Pass these parameters to your processing functions as needed
    process_homework_dirs(data_dir, similarity_thresholds, min_block_size, question_filter_threshold)
    # After processing all homework directories
    generate_enhanced_global_report(data_dir)

