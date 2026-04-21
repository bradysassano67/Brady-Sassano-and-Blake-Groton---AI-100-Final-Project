Fake News Detection Using Deep Learning

Overview
This project builds a deep learning model to classify news articles as Real or Fake. The goal is to detect misinformation using a Bidirectional LSTM neural network trained on the WELFake dataset.

This was completed as a midterm project for AI 100.

Problem Definition
The task is a binary text classification problem.
Input: News article title and body text
Output: 1 = Real, 0 = Fake

Misinformation spreads rapidly online. Automated detection systems help identify unreliable content at scale.

Dataset

Primary dataset: WELFake_Dataset.csv

After cleaning:

Total samples: 72,079

Real (1): 37,051

Fake (0): 35,028

Preprocessing steps:

Combined title + text

Lowercased text

Removed URLs

Removed punctuation

Removed extra whitespace

Removed extremely short entries

Data was split 80 percent training and 20 percent testing using stratified sampling.

Model Architecture

The model uses:

Embedding layer
Bidirectional LSTM (64 units)
Dense layer (ReLU activation)
Dropout (0.4)
Output layer (Sigmoid activation)

Loss function: Binary Cross Entropy
Optimizer: Adam
Batch size: 64
Epochs: up to 6 with early stopping

Results

Test Accuracy: 0.9541
Test Loss: 0.1145

Confusion Matrix:

[[6790, 216],
[ 446, 6964]]

Performance Summary:

Fake class
Precision: 0.9384
Recall: 0.9692
F1-score: 0.9535

Real class
Precision: 0.9699
Recall: 0.9398
F1-score: 0.9546

The model performs well on both classes with balanced performance and low misclassification rates.

Project Structure

project-1-fake-news/
│
├── src/
│ └── train_lstm.py
│
├── outputs/
│ ├── accuracy_curve.png
│ ├── loss_curve.png
│ ├── confusion_matrix.png
│
├── data/
│ └── WELFake_Dataset.csv
│
├── requirements.txt
├── report.pdf
└── README.md

How to Run

Clone the repository

git clone YOUR_REPOSITORY_URL

Navigate into the project folder

cd project-folder-name

Create virtual environment

python3 -m venv .venv
source .venv/bin/activate

Install dependencies

python -m pip install -r requirements.txt

Run training

python src/train_lstm.py

Note: The trained model file (.keras) is not included due to GitHub file size limits.
The model can be reproduced by running the training script.
