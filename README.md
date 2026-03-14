
# Resume Classification using NLP and Machine Learning

## Project Overview

This project builds an intelligent resume classification system that automatically predicts the most relevant job role from a resume. The system uses Natural Language Processing and Machine Learning techniques to analyze resume text and classify it into predefined job categories.

The application allows users to upload a resume in PDF or DOCX format or paste resume text directly. The system processes the content, extracts meaningful information, and predicts the most suitable job role.

This project demonstrates how machine learning can assist in automating the resume screening process in recruitment systems.

Live Demo [https://resume-classification-using-nlp-and-machine-learning-jj8kkx7gw.streamlit.app/]

---

# Features

• Upload resume in PDF or DOCX format
• Paste resume text manually
• Automatic text extraction from documents
• Text preprocessing using NLP techniques
• Resume classification using a trained machine learning model
• Visualization of prediction confidence
• Display of alternative possible job matches
• Interactive web interface built with Streamlit

---

# Technologies Used

Python

Machine Learning
Natural Language Processing (NLP)

Libraries used

• Scikit-learn
• NLTK
• Streamlit
• Joblib
• pdfplumber
• python-docx
• Matplotlib
• NumPy
• Regex

---

# Machine Learning Workflow

The system follows the following pipeline:

1 Data Collection
Resume datasets containing different job categories are collected.

2 Text Preprocessing
Resume text is cleaned using several NLP techniques such as:

• Lowercasing
• Removing special characters
• Removing stopwords
• Lemmatization

3 Feature Extraction
The cleaned text is converted into numerical features using **TF-IDF Vectorization**.

4 Model Training
A **Support Vector Classifier (SVC)** is trained to classify resumes based on the extracted features.

5 Prediction
The trained model predicts the most suitable job role for a given resume.

---

# Job Categories

The model is currently trained to classify resumes into the following roles:

• React Developer
• SQL Developer
• Workday Developer
• PeopleSoft Developer

The system can easily be extended by training it on more resume datasets.

---

# Project Structure

```
Resume-Classification/
│
├── app.py                # Streamlit application
├── tfidf.pkl             # TF-IDF vectorizer
├── SVCmodel.pkl          # Trained machine learning model
├── resume_classification.ipynb   # Model training notebook
├── requirements.txt      # Required dependencies
└── README.md             # Project documentation
```

---

# Installation

Clone the repository

```
git clone https://github.com/yourusername/resume-classification.git
```

Move to project directory

```
cd resume-classification
```

Install required libraries

```
pip install -r requirements.txt
```

---

# Running the Application

Run the Streamlit application

```
streamlit run app.py
```

The web application will start and open in your browser.

---

# How the Application Works

1 User uploads a resume or pastes resume text
2 The system extracts the text from the document
3 The text is cleaned using NLP preprocessing
4 The cleaned text is transformed using TF-IDF vectorization
5 The trained SVC model predicts the most relevant job role
6 The application displays

• Predicted job role
• Model confidence
• Alternative possible matches

---

# Example Output

Input: Resume containing skills like React, JavaScript, frontend development

Predicted Role: React Developer

The system also displays relative confidence scores for other possible roles.

---



---

