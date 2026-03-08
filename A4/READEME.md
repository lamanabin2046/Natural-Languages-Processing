# A4: BERT Pretraining & Sentence Similarity Application

This repository contains the implementation of **BERT-style pretraining tasks**, **natural language inference (NLI) classification**, and a **sentence similarity web application** developed for **NLP Assignment 4**.

The project covers:
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- NLI classification (Entailment / Neutral / Contradiction)
- Sentence similarity using cosine similarity
- Deployment of the trained model via a Dash web application

---

## Project Folder Structure

A4/
├── artifacts/ # Training artifacts and intermediate outputs
├── data/ # Dataset files (processed / cached)
├── model/
│ ├── bert_model.pth # Trained BERT model
│ ├── sen_bert.pth # Sentence-BERT model
├── code/
│ ├── app.py # Dash web application for sentence similarity
│ ├── Bert.py # BERT model architecture and helper functions
│ ├── pycache/
│ └── .ipynb_checkpoints/
├── a4-task1.ipynb # Task 1: MLM and NSP pretraining
├── a4-task-2-3-4.ipynb # Tasks 2, 3, 4: Training, Evaluation, and Similarity
├── bert.py # Standalone BERT utilities
├── NLP_2026_A4_Do_you_AGREE.pdf # Assignment description
├── pycache/
└── .ipynb_checkpoints/


---

## Task Description

### Task 1: BERT Pretraining (MLM and NSP)
Implemented from scratch using PyTorch:
- Tokenization and input construction
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)
- Custom batch generation
- Model training

Dataset used:
- WikiText-103 (via HuggingFace `datasets`)

---

### Task 2: Model Training
- Train, validation, and test split
- Adam optimizer
- GPU support (CUDA if available)
- Loss monitoring for MLM and NSP

---

### Task 3: Model Evaluation (NLI Classification)
The trained model is evaluated on a **Natural Language Inference (NLI)** task with three classes:
- Entailment
- Neutral
- Contradiction

---

### Task 4: Sentence Similarity
- Sentence embeddings generated using trained BERT
- Cosine similarity used for semantic comparison
- Evaluation on similar and dissimilar sentence pairs

---

## Classification Report (NLI)

The following classification report summarizes the performance of the model on **1000 NLI samples**:

| Class | Precision | Recall | F1-score | Support |
|------|----------|--------|----------|---------|
| Entailment | 0.33 | 0.07 | 0.12 | 338 |
| Neutral | 0.33 | 0.92 | 0.48 | 328 |
| Contradiction | 0.33 | 0.01 | 0.02 | 334 |
| Accuracy |  |  | 0.33 | 1000 |
| Macro Avg | 0.33 | 0.33 | 0.21 | 1000 |
| Weighted Avg | 0.33 | 0.33 | 0.20 | 1000 |

### Interpretation
- The model strongly favors the Neutral class, as shown by high recall.
- Very low recall for Entailment and Contradiction indicates insufficient fine-tuning or class bias.
- Overall accuracy (33 percent) is close to a random baseline for a three-class classification problem.
- The model performs better for sentence similarity than for NLI classification without further task-specific training.

---

## Web Application (`code/app.py`)

The `app.py` file implements an interactive sentence similarity web application using Dash.

Features:
- User inputs two sentences
- Sentence embeddings generated using the trained model
- Cosine similarity computation
- Real-time similarity score display
- Automatic GPU or CPU selection

Purpose:
- Demonstrates practical deployment of the trained NLP model
- Bridges theoretical modeling and real-time application

---

## Sentence Similarity Results

| Metric | Value |
|------|------|
| Training Loss | 0.918 |
| Cosine Similarity (Similar Sentences) | 0.992 |
| Cosine Similarity (Dissimilar Sentences) | 0.999 |

---

Demo:
## DEMO
![Do You Agree](demo.gif)

## How to Run

### Step 1: Install Dependencies
```bash
pip install torch datasets dash numpy scikit-learn transformers
Step 2: Run Notebooks
jupyter notebook
Execute in order:

a4-task1.ipynb

a4-task-2-3-4.ipynb

Step 3: Run Web Application
cd code
python app.py
Open the local URL shown in the terminal.

Learning Outcomes
Understanding BERT architecture

Implementing MLM and NSP

NLI classification using BERT

Sentence embedding and semantic similarity

PyTorch-based training pipelines

Deploying NLP models using Dash

Author
Nabin Gangtan Lama
MSc in Data Science & Artificial Intelligence
Asian Institute of Technology (AIT) 

