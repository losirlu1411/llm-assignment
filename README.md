# llm-assignment
# Sentiment Analysis with BERT on IMDb Movie Reviews 🎬📊

This project demonstrates how to fine-tune a BERT-based language model to classify movie reviews from the IMDb dataset as either **positive** or **negative**. The notebook includes preprocessing, tokenization, model training, evaluation, and inference using Hugging Face Transformers.

---

##  Project Structure

- `llm assignment.ipynb`: Jupyter Notebook containing the full code for training and evaluating the sentiment classification model.
- `README.md`: Project overview and instructions (this file).

---

##  Dataset

- **Name**: IMDb Movie Reviews  
- **Source**: [Hugging Face Datasets](https://huggingface.co/datasets/imdb)  
- **Description**: 50,000 labeled movie reviews — 25,000 for training and 25,000 for testing (binary sentiment: 0 = negative, 1 = positive).

---

## Model

- **Base Model**: `bert-base-uncased`
- **Library**: Hugging Face Transformers
- **Fine-tuning**: 3 epochs using the `Trainer` API
- **Evaluation Metric**: Accuracy (and F1-score via `classification_report`)

---

## Key Steps

1. Load IMDb dataset and clean it using Pandas
2. Perform Exploratory Data Analysis (EDA)
3. Tokenize data using `BertTokenizer`
4. Fine-tune BERT with Hugging Face `Trainer`
5. Evaluate results with classification report and confusion matrix
6. Predict sentiment on new text samples

---

Example prediction:
> _"Completely blown away by the performances and direction."_ → **Positive**

---

## Getting Started

### Install Dependencies

```bash
pip install transformers datasets evaluate wordcloud matplotlib seaborn
