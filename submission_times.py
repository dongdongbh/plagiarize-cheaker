"""
Plagiarism Detection Tool

Author: Dongda Li
Email: dli160@syr.edu

This file is part of the Plagiarism Detection Tool project.
This project is licensed under the MIT License - see the LICENSE file for details.
"""

import os
import pandas as pd
import re

def parse_filename(filename):
    # Extract homework number, user ID, and timestamp from the filename
    match = re.search(r'Homework (\d+)_([^_]+)_attempt_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_', filename)
    if match:
        hw_number = int(match.group(1))  # Convert to int to sort numerically later
        user_id = match.group(2)
        timestamp = match.group(3).replace('-', ':').replace(':', '/', 2).replace(':', '-')
        return hw_number, user_id, timestamp
    return None

def collect_submission_times(data_dir):
    # Dictionary to hold the data
    submissions = {}

    # List all homework directories
    hw_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('hw')]

    for hw_dir in hw_dirs:
        # Path to the homework directory
        hw_path = os.path.join(data_dir, hw_dir)
        
        # List all files in the homework directory
        for filename in os.listdir(hw_path):
            if (filename.endswith('.pdf') or filename.endswith('.docx')) and 'cover' not in filename.lower():
                parsed = parse_filename(filename)
                if parsed:
                    hw_number, user_id, timestamp = parsed
                    if user_id not in submissions:
                        submissions[user_id] = {}
                    # Update the dictionary only if this timestamp is later or if the homework entry doesn't exist yet
                    if hw_number not in submissions[user_id] or submissions[user_id][hw_number] < timestamp:
                        submissions[user_id][hw_number] = timestamp

    return submissions

def generate_spreadsheet(submissions, output_file):
    # Create a DataFrame from the submissions dictionary
    df = pd.DataFrame.from_dict(submissions, orient='index')
    df.sort_index(inplace=True)  # Sort by student ID

    # Sort the columns by homework number (assuming they are all integers)
    sorted_columns = sorted(df.columns, key=lambda x: int(x))
    df = df[sorted_columns]  # Reorder the columns based on sorted homework numbers

    df.to_csv(output_file)

if __name__ == "__main__":
    data_dir = './data/'  # Path to the data directory
    submissions = collect_submission_times(data_dir)
    generate_spreadsheet(submissions, './data/submission_times.csv')

