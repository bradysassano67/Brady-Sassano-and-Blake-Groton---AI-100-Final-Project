# AI-100 Final Project – Bug Analysis

## Overview

This project is based on our AI-100 midterm project, where we built a fake news classifier using a Bidirectional LSTM model. The system classifies news articles as **Fake (0)** or **Real (1)** using text data from a labeled dataset.

For the final project, we reused this system and intentionally introduced bugs into different parts of the pipeline. The goal was to analyze how errors affect the system and improve debugging reasoning using feedback from a large language model (GenAI).

---

## Final Project Focus

The main objective of this assignment was:

* To introduce **intentional bugs** into the AI system
* To write **initial self-reflections** explaining the issue
* To use **GenAI feedback** to improve those reflections
* To better understand how to debug machine learning systems

Detailed bug cases and analysis are documented in the submitted **Google Sheet** and **PDF report**.

---

## Model Description

The model used in this project includes:

* **Embedding Layer** – converts words into dense vectors
* **Bidirectional LSTM Layer** – captures context from both directions
* **Dropout Layer** – reduces overfitting
* **Dense Output Layer (Sigmoid)** – outputs probability for binary classification

The model is trained using:

* Binary Cross-Entropy Loss
* Adam Optimizer

---

## Data Processing

The dataset consists of labeled news articles. The following preprocessing steps are applied:

* Convert text to lowercase
* Remove URLs and unnecessary punctuation
* Combine title and article text
* Tokenize words into sequences
* Pad/truncate sequences to a fixed length

---

## Project Structure

```
AI-100-Final-Project/
├── Microsoft Excel PDF
├── Microsoft Word Report PDF
├── train_lstm.py
├── requirements.txt
├── tokenizer.json
├── settings.json
├── run_metadata.json
├── accuracy_curve.png
├── loss_curve.png
├── confusion_matrix.png
├── Screenshot 2026-03-01 at 8.06.14 PM.png
└── README.md
```

---

## How to Run

1. Install dependencies:

```
python -m pip install -r requirements.txt
```

2. Run the training script:

```
python train_lstm.py
```

---

## Notes

* The trained model file (`.keras`) is not included due to GitHub file size limits.
* The dataset is also not included and can be downloaded from Kaggle.
* The model can be reproduced by running the training script.

---

## Learning Outcomes

Through this project, we learned:

* How small changes in code can significantly impact model performance
* How to identify and reason about bugs in machine learning systems
* How GenAI can assist in improving debugging and reflection skills

---

## Author

Brady Sassano and Blake Groton

