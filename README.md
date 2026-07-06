# Topic Classification of Russian News (Lenta.ru, 100k documents).

This project implements a full, reproducible NLP pipeline for classifying Russian news articles by topic using classical embedding‑based methods. It includes data loading, reprocessing, tokenization, vectorization using multiple embedding strategies, model training, evaluation, and comparison of embedding quality.

## 1. Project Overview.

This repository contains an end‑to‑end text classification pipeline built on a 100,000‑document subset of the Lenta.ru news corpus.
The goal is to evaluate how far classical embedding‑based NLP methods (Word2Vec, Navec, RusVectōrēs, TF‑IDF‑weighted embeddings) can go on a large real‑world Russian dataset.

The pipeline includes:

 - efficient preprocessing of Russian text;
 - stratified splitting (60/20/20);
 - training custom Word2Vec embeddings;
 - loading pretrained Navec and RusVectōrēs embeddings;
 - building sentence embeddings (mean pooling);
 - TF‑IDF‑weighted pooling;
 - Logistic Regression training;
 - intrinsic evaluation of embeddings;
 - final evaluation on a held‑out test set.

## 2. Dataset.

- source: Lenta.ru news dataset (Corus / Kaggle);

- size used: 100,000 documents;

- fields: title, text, topic;

- rare classes removed (<2 samples) to ensure valid stratification.

Each record represents a single news article with three fields:
 - title,
 - text,
 - topic.

## 3. Preprocessing Pipeline.

The preprocessing pipeline includes:

 - lowercasing;
 - removing non‑alphabetic characters;
 - normalization of whitespace;
 - simple tokenization (for Word2Vec / Navec);
 - morphological tokenization (lemma_POS) for RusVectōrēs;
 - removing empty texts;
 - filtering out classes with <2 samples.

Why morphological tokenization?

RusVectōrēs embeddings are trained on lemma_POS tokens (e.g., москва_PROPN, идти_VERB). Using the same format significantly improves embedding quality.

## 4. Train/Validation/Test Split.

- train: 60%,
- validation: 20%,
- test: 20%,
- stratified by topic,
- random_state = 42 for reproducibility.

## 5. Embedding Models Evaluated.

The project compares four embedding strategies:

- Word2Vec (trained on Lenta.ru corpus)

- Navec (pretrained Russian embeddings)

- RusVectōrēs (lemma_POS embeddings)

- TF‑IDF‑weighted embeddings (best performing)

### Validation macro F1 results:

 - Navec — 0.630 (best)

 - TF‑IDF‑weighted embeddings — 0.625

 - Word2Vec — 0.614

 - RusVectōrēs (with morphology) — 0.566

Intrinsic evaluation includes:

 - nearest neighbors;
 - odd‑one‑out;
 - semantic similarity checks.

Examples saved in:

reports/intrinsic_examples.txt

## 5.1 Word2Vec Hyperparameters Justification.

The Word2Vec model was trained on the Lenta.ru corpus using a set of hyperparameters chosen to balance quality, training stability, and computational efficiency.

*vector_size = 100*

A 100‑dimensional embedding space provides an optimal balance for a medium‑sized corpus:

- Lenta.ru contains ~100k documents — not large enough to benefit from 300‑dimensional vectors.

- Higher dimensions require more data and training time.

- Empirically, 100 dimensions capture semantic relations well for Russian news.

*window = 5*

Defines how many words to the left and right are considered context:

- A window of 5 works well for Russian morphology.

- Captures meaningful semantic relations in news articles.

- Smaller windows capture mostly syntax; larger windows dilute context.

*min_count = 5*

Minimum frequency threshold for a word to be included in the vocabulary:

- Rare words introduce noise and unstable gradients.

- A threshold of 5 removes garbage tokens while preserving important entities (names, cities, organizations).

*workers = 4*

Number of parallel threads:

- Speeds up training.

- Suitable for typical environments (local machine, Colab).

- Avoids excessive CPU load.

*sg = 1 (skip‑gram)*

Model architecture:

- sg = 1 → skip‑gram, better for rare words.

- sg = 0 → CBOW, faster but worse on infrequent tokens.

- News contain many rare entities (names, organizations), so skip‑gram produces higher‑quality embeddings.

*epochs = 10*

Number of training passes over the corpus:

- Fewer epochs lead to underfitting.

- More epochs give diminishing returns on a medium corpus.

- 10 epochs provide stable convergence and good semantic quality.

*seed = RANDOM_STATE*

A fixed seed ensures reproducibility:

- Required for consistent experimental results.

- Guarantees identical embeddings across repeated runs.

## 6. Sentence Embedding Strategies.

Two main strategies were evaluated:

### Mean Pooling

Simple average of word vectors.
Works well for Word2Vec, Navec, RusVectōrēs.

### TF‑IDF‑Weighted Pooling

Each word vector is multiplied by its TF‑IDF weight:

<img width="163" height="60" alt="image" src="https://github.com/user-attachments/assets/087462ec-7bbd-43b5-95cf-af2f07a0be3e" />

This method significantly improves embedding quality and nearly matches Navec.

## 7. Model Training.

All models use:

- Logistic Regression

- max_iter = 1000

- n_jobs = -1

- macro F1 as the main metric

Validation metrics for all models are saved in:

reports/dz2_metrics.txt

## 8. Final Evaluation (Test Set).

The best model (TF‑IDF‑weighted embeddings) is evaluated on the held‑out test set.

Metrics include: 
 - Accuracy;
 - Macro F1;
 - Classification Report;
 - Confusion Matrix.

Confusion matrix saved in:

reports/confusion_matrix.png

Strong classes: Russia, World, Culture, Sport, Economy
Weak classes: Library (1 sample), Science & Technology (54 samples)

Macro F1 = 0.63 reflects the difficulty of rare classes.

## 9. Error Analysis.

Typical misclassification patterns include:

 - World → Russia,
 - Russia → World,
 - Economy → Russia,
 - Lifestyle → World.

These confusions arise due to overlapping vocabulary and similar contextual cues across topics — a known limitation of classical embedding‑based models.

## 10. Possible Improvements.

If extended into a production pipeline, the following enhancements could be explored:

Embedding Improvements

- FastText or transformer embeddings

- contextual embeddings (RuBERT, mBERT)

Feature Engineering

- char‑level n‑grams

- text length features

- named entity indicators

Model Improvements

- LinearSVC

- class balancing (oversampling, class weights)

- targeted augmentation for rare classes

## 11. How to Run.

Install dependencies

pip install -r requirements.txt

Run notebooks in order

- 01_data_preparation.ipynb

- 02_embeddings_training.ipynb

- 03_tfidf_weighting_and_evaluation.ipynb

Evaluate the best model

python src/evaluate.py

Experiment Report

Detailed model evaluation and intrinsic analysis are available in:

reports/dz2_metrics.txt,

reports/intrinsic_examples.txt.
