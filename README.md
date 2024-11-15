# SciKey: Advanced Key Phrase Extraction and Summarization for Scientific Research

![SciKey Logo](logo.png)

## Overview

SciKey is an advanced Natural Language Processing (NLP) tool designed for extracting key phrases and summarizing scientific research papers. This tool aims to help researchers and academics quickly identify essential information from extensive scientific documents, enhancing productivity and simplifying the research process.

## Features

- **Key Phrase Extraction**: Automatically extract significant key phrases from scientific texts.
- **Summarization**: Generate concise summaries of research papers.
- **User-Friendly Interface**: Easy-to-use interface for uploading and processing documents.
- **Customization**: Options for customizing the length and detail level of the summaries.
- **Enhanced Information Retrieval**: Integrates Topic Modeling with Question-Answering Systems to provide exhaustive information on a given topic from a set of academic articles.

## Installation

To use SciKey, you need to have Python installed on your machine. Follow the steps below to set up the project:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/Deeraj7/scikey.git
    cd scikey
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the application:**
    ```sh
    python app.py
    ```

## Usage

1. **Upload Document**: Upload your scientific document (in .pdf, .docx, or .txt format) through the user interface.
2. **Extract Key Phrases**: Click the "Extract Key Phrases" button to get a list of key phrases from the document.
3. **Generate Summary**: Click the "Generate Summary" button to get a concise summary of the document.

## Files

- `app.py`: Main application file that runs the SciKey tool.
- `requirements.txt`: List of Python packages required to run the application.
- `random_data.csv`: Sample data file used for testing the key phrase extraction and summarization algorithms.
- `image.png`: Logo or image used in the project.

## Dataset

The dataset used in this project is the COVID-19 Open Research Dataset (CORD-19) provided by the Allen Institute for AI. Due to its large size, it is not included in this repository. You can access it through the following links:

- [COVID-19 Open Research Dataset (CORD-19) GitHub](https://github.com/allenai/cord19)
- [Sample Data Link](https://drive.google.com/file/d/1eClGP2AnbomBfxHWhy0nL0pw7pszglTF/view?usp=sharing)
- [Full Dataset on Kaggle](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)

## Methodology

The project integrates Topic Modeling with Question-Answering Systems to provide enhanced information retrieval from research articles. Key techniques used include:

- **Topic Modeling**: LDA, NMF, BERTopic, Correx
- **Question-Answering System**: Based on TF-IDF techniques

### Data Pre-Processing

1. Loading the dataset into a pandas DataFrame.
2. Text cleaning (lowercasing, removing special characters, removing stopwords).
3. Combining cleaned titles and abstracts into a unified text corpus.
4. Transforming text into a TF-IDF matrix using `TfidfVectorizer`.

### Topic Modeling

- **Non-negative Matrix Factorization (NMF)**: Applied to the TF-IDF matrix to extract topics and keywords.

### Question-Answering System

1. Processing user queries.
2. Finding relevant articles or excerpts.
3. Extracting and presenting answers.

## Contributing

We welcome contributions to enhance SciKey. Please follow the steps below to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

SciKey is released under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or feedback, please contact **Deeraj Thakkilapati** at thakkilapatideeraj@gmail.com
