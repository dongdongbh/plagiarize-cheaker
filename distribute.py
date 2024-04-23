"""
Plagiarism Detection Tool

Author: Dongda Li
Email: dli160@syr.edu

This file is part of the Plagiarism Detection Tool project.
This project is licensed under the MIT License - see the LICENSE file for details.
"""

import os
import shutil
import pandas as pd
from math import ceil

def distribute_files(data_dir, num_staff):
    # Load the CSV file with student IDs
    student_csv_path = os.path.join(data_dir, 'global-report_S.csv')
    df = pd.read_csv(student_csv_path)
    
    # Assume the relevant column for student IDs is named 'Student ID'
    student_ids = df['Student ID'].tolist()

    # Number of students per staff, considering the last part may have fewer students
    students_per_staff = ceil(len(student_ids) / num_staff)

    # Gather all files belonging to the student IDs
    student_files = {student_id: [] for student_id in student_ids}
    homework_dirs = [d for d in os.listdir(data_dir) if d.startswith('hw')]
    
    # Collect files for each student
    for hw_dir in homework_dirs:
        hw_path = os.path.join(data_dir, hw_dir)
        for file in os.listdir(hw_path):
            if file.endswith(('.pdf', '.docx')):  # Ensure only PDF and DOCX files are considered
                parts = file.split('_')
                student_id = parts[1]
                if student_id in student_ids:
                    student_files[student_id].append(os.path.join(hw_path, file))

    # Create directories for each part and distribute files
    for i in range(num_staff):
        part_start = i * students_per_staff
        part_end = (i + 1) * students_per_staff
        part_students = student_ids[part_start:part_end]
        
        part_dir = os.path.join(data_dir, f'part{i+1}')
        os.makedirs(part_dir, exist_ok=True)
        
        for student_id in part_students:
            student_dir = os.path.join(part_dir, f'{student_id}')
            os.makedirs(student_dir, exist_ok=True)
            
            # Copy files to the student's directory
            for file_path in student_files[student_id]:
                shutil.copy(file_path, student_dir)

if __name__ == "__main__":
    data_dir = './data'  # Path to the data directory
    num_staff = 3  # Number of course staff members
    distribute_files(data_dir, num_staff)

