import argparse
import pandas as pd
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
import re
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

def load_student_dict(data_dir):
    filepath = os.path.join(data_dir, "students-list.csv")
    # Load the CSV file
    df = pd.read_csv(filepath)
    
    # Assuming the columns are named 'Student ID', 'First Name', 'Last Name'
    # Create a new column for full name by combining first and last names
    df['Full Name'] = df['First Name'].str.strip() + ' ' + df['Last Name'].str.strip()
    
    # Create a dictionary mapping user names (student IDs) to full names
    student_dict = df.set_index('Username')['Full Name'].to_dict()    
    return student_dict

def update_detailed_cheating_instances(user1, user2, hw_number, similarity, collaborators_listed):
    # This function now also receives a boolean `collaborators_listed` indicating if collaborators were listed
    if user1 not in detailed_cheating_instances:
        detailed_cheating_instances[user1] = []

    detailed_cheating_instances[user1].append((hw_number, user2, similarity, collaborators_listed))

def generate_enhanced_global_report(data_dir):
    global_csv_path = os.path.join(data_dir, 'global-report.csv')

    print(f"Generating enhanced global report to file {global_csv_path}")
    
    with open(global_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Student ID', 'Cheating Frequency (Unique Assignments)', 'Max Similarity', 'Average Similarity', 'Details', 'Unlisted Collaborators'])

        sorted_students = sorted(detailed_cheating_instances.items(), key=lambda x: len(set(hw_number for hw_number, _, _, _ in x[1])), reverse=True)

        for student_id, details in sorted_students:
            hw_details = {}
            hw_unlisted_collaborators = set()  # Track homeworks with unlisted collaborators using a set
            
            for hw_number, peer_id, similarity, collaborators_listed in details:
                if hw_number not in hw_details:
                    hw_details[hw_number] = []
                hw_details[hw_number].append((peer_id, similarity))
                
                if not collaborators_listed:
                    hw_unlisted_collaborators.add(hw_number)  # Add homework number if collaborators were not listed
            
            details_str = "; ".join(
                f"HW{hw}: {', '.join(f'{peer_id} (Similarity: {similarity:.2%})' for peer_id, similarity in peers)}"
                for hw, peers in sorted(hw_details.items())
            )

            max_similarity = max(similarity for _, similarity in sum(hw_details.values(), []))
            avg_similarity = sum(similarity for _, similarity in sum(hw_details.values(), [])) / sum(len(peers) for peers in hw_details.values())

            unlisted_collaborator_times = len(hw_unlisted_collaborators)  # Number of homeworks where no collaborators were listed

            writer.writerow([student_id, len(hw_details), f"{max_similarity:.2%}", f"{avg_similarity:.2%}", details_str, unlisted_collaborator_times])


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


def append_to_hw_csv(hw_csv_path, data):
    with open(hw_csv_path, 'a', newline='') as file:
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


def sentence_filter(text, junk_sentences, threshold):
    """Filter out specified sentences from text based on a similarity threshold."""
    filtered_sentences = []
    doc = nlp(text)
    for sent in doc.sents:
        if not any(difflib.SequenceMatcher(None, str(sent), junk).ratio() > threshold for junk in junk_sentences):
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

def extract_collaborators(text, pattern):
    """ Extract sentences containing collaborator names using a regex pattern. """
    sentences = text.split('.')
    collaborators = set()
    for sentence in sentences:
        if pattern.search(sentence):
            collaborators.update(pattern.findall(sentence))
    return collaborators

def extract_and_filter_text(filepath, question_sentences, filter_threshold, student_names, current_student):
    """Extract text and filter based on provided content, then extract potential collaborators."""
    text = None
    filtered_text = None
    collaborators = set()

    if filepath.endswith('.pdf'):
        text = pdf_to_text(filepath)
    elif filepath.endswith('.docx'):
        text = docx_to_text(filepath)
    else:
        return filepath, None, set()  # Unsupported format

    if text:
        # Extract collaborator names from the text
        names_to_match = [name for name in student_names.values() if name != student_names[current_student]]
        pattern = re.compile('|'.join(re.escape(name) for name in names_to_match), re.IGNORECASE)
        collaborators = extract_collaborators(text, pattern)

        # Filter text if not a cover letter
        if 'cover' not in os.path.basename(filepath).lower():
            filtered_text = sentence_filter(text, question_sentences, filter_threshold)
        else:
            filtered_text = None  # Do not use cover letter text for plagiarism checks

    return filepath, filtered_text, collaborators

def process_pdfs(data_dir, filter_sentences, question_filter_threshold, student_names):
    """Process all PDFs and DOCX files for text extraction, filtering, and collaborator extraction."""
    files_to_process = [
        (os.path.join(data_dir, filename), filter_sentences, question_filter_threshold, student_names, filename.split('_')[1])
        for filename in os.listdir(data_dir) 
        if (filename.endswith('.pdf') or filename.endswith('.docx'))
    ]

    print("Extracting text and checking for collaborators...")
    with Pool(cpu_count()) as pool:
        # Use starmap to correctly pass multiple arguments from each tuple
        results = pool.starmap(extract_and_filter_text, files_to_process)

    texts = {}
    submission_paths = {}
    collaborator_mentions = {}
    for filepath, text, collaborators in results:
        current_student = filepath.split('_')[1]
        submission_paths[current_student] = filepath
        if text:
            texts[current_student] = text
        if collaborators:
            collaborator_mentions[current_student] = collaborators

    return texts, submission_paths, collaborator_mentions


def process_single_homework(data_dir, question_path, cover_letter_path, similarity_threshold, min_block_size, question_filter_threshold, hw_number, student_names):
    hw_csv_path = os.path.join(data_dir, f'report_hw{hw_number}.csv')
    with open(hw_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Student ID 1', 'Student ID 2', 'Assignment Number', 'Similarity Score', 'Matched Sections', 'Explanatory Note', 'Collaborator u1', 'Collaborator u2'])

    question_content = get_sentences(question_path, nlp)
    cover_letter_content = get_sentences(cover_letter_path, nlp)
    combined_filter_sentences = question_content + cover_letter_content

    texts, submission_paths, collaborator_data = process_pdfs(data_dir, combined_filter_sentences, question_filter_threshold, student_names)

    pairs = generate_comparison_pairs(texts)
    comparison_data = [(user1, text1, user2, text2, similarity_threshold, min_block_size) for user1, text1, user2, text2 in pairs]

    print("Comparing data...")
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(compare_texts, comparison_data), total=len(comparison_data)))

    filtered_results = [result for result in results if result is not None]

    for user1, user2, similarity, matched_sections in filtered_results:
        collaborators1 = ", ".join(collaborator_data.get(user1, []))
        collaborators2 = ", ".join(collaborator_data.get(user2, []))
        collaborators_listed1 = bool(collaborators1)
        collaborators_listed2 = bool(collaborators2)

        data = [user1, user2, hw_number, similarity, "; ".join(matched_sections), "Potential plagiarism detected.", collaborators1, collaborators2]
        append_to_hw_csv(hw_csv_path, data)
        update_detailed_cheating_instances(user1, user2, hw_number, similarity, collaborators_listed1)
        update_detailed_cheating_instances(user2, user1, hw_number, similarity, collaborators_listed2)
        handle_individual_folders(data_dir, user1, hw_number, user2, similarity, matched_sections, submission_paths[user1], submission_paths[user2])
        handle_individual_folders(data_dir, user2, hw_number, user1, similarity, matched_sections, submission_paths[user2], submission_paths[user1])


    # Print summary statistics for this homework
    print_summary_statistics(filtered_results)

def print_summary_statistics(filtered_results):
    involved_students = {user for result in filtered_results for user in result[:2]}
    print(f"Total document pairs with more than specified similarity: {len(filtered_results)}")
    print(f"Total number of unique students involved: {len(involved_students)}")

def process_homework_dirs(data_dir, config):
    # Extract necessary configurations
    similarity_thresholds = config["similarity_thresholds"]
    min_block_size = config["min_block_size"]
    question_filter_threshold = config["question_filter_threshold"]
    specified_homeworks = config.get("specified_homeworks", None)
    student_id_name_dict = load_student_dict(data_dir)

    # Identify all hw directories
    hw_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('hw')]

    # Filter hw_dirs based on specified_homeworks, if any
    if specified_homeworks is not None:
        hw_dirs = [d for d in hw_dirs if int(d[2:]) in specified_homeworks]

    cover_letter_path = os.path.join(data_dir, "questions", "675cover.pdf")
    
    for hw_dir_name in hw_dirs:
        hw_number = hw_dir_name[2:]  # Extract the numeric part
        current_assignment = f"HW{hw_number}"
        current_threshold = similarity_thresholds.get(current_assignment, 0.15)

        question_path = os.path.join(data_dir, "questions", f"Homework{hw_number}.pdf")
        print(f"------------------Processing {hw_dir_name} with questions from {question_path}--------------------")
        process_single_homework(os.path.join(data_dir, hw_dir_name), question_path, cover_letter_path, current_threshold, min_block_size, question_filter_threshold, hw_number, student_id_name_dict)

if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_file)

    process_homework_dirs(config["data_dir"], config)
    # After processing all homework directories
    generate_enhanced_global_report(config["data_dir"])

