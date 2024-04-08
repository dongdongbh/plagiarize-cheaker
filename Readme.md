# Plagiarism Detection Tool for PDF Documents

## Overview

This project provides a simple yet effective tool for detecting potential plagiarism in text-based PDF documents. Utilizing Python, the tool first converts PDF files to text and then performs similarity comparisons between all pairs of documents within a specified directory. This approach can be particularly useful for educators, teaching assistants, and content managers aiming to ensure the originality of submitted texts.

## Features

- **PDF to Text Conversion**: Extracts plain text from PDF documents for analysis.
- **Similarity Checking**: Compares the textual content of each pair of documents to determine similarity scores.
- **Scalable**: Easily adaptable to handle large sets of documents.

## Getting Started

### Prerequisites

- Python 3.x
- Installation of required Python libraries: `PyMuPDF`, `difflib`

### Installation

1. Clone this repository to your local machine.

```
git clone https://github.com/dongdongbh/plagiarize-cheaker.git
```

2. Navigate to the cloned repository directory.

3. Install the required Python libraries.


### Usage

1. Place all PDF documents you wish to analyze in the `./data/` directory.
2. Run the script to extract text from PDFs and compare documents for similarity.
`python main.py`

3. Review the output similarity scores to identify potential plagiarism.

![Result](./img/result.png "Result of similarity comparison between PDF documents")

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This tool was inspired by the need to automate the plagiarism detection process in educational settings.
- Thanks to the contributors of PyMuPDF for providing a robust library for PDF processing.

