# Math Question Classification using Classical Machine Learning

## 1. Problem Overview

The objective of this project is to build a classical machine learning system that can automatically classify high school mathematics questions into their respective subtopics such as Algebra, Geometry, Precalculus, Number Theory, and related areas.

Each question is provided as an individual JSON file, and the classification task is framed as a multi-class text classification problem based solely on the problem statement.

---

## 2. Dataset Structure and Assumptions

The dataset is organized into two main directories: `train` and `test`.  
Each of these directories contains subfolders named after math subtopics (e.g., `algebra`, `geometry`, `precalculus`).

Each subfolder contains multiple JSON files, where:
- the `problem` field contains the math question text
- additional fields such as `solution`, `level`, and `type` are present

The folder names were treated as the ground-truth labels for classification.

To prevent data leakage, **only the problem text** was used as input features. The solution fields were intentionally excluded from the model pipeline.

---

## 3. Data Loading and Preprocessing

All JSON files were read dynamically by traversing the directory structure.  
Each problem statement was extracted and paired with its corresponding label based on the folder name.

Minimal preprocessing was applied:
- text was lowercased
- no manual token filtering or mathematical symbol removal was performed

This decision was made to allow the vectorization methods to naturally learn which tokens are informative for topic classification.

---

## 4. Feature Representation

Two different text vectorization strategies were explored:

### 4.1 Bag of Words (BoW)
The Bag of Words approach represents text as raw token frequency counts. While simple and effective, it treats all tokens as equally important.

### 4.2 TF-IDF
TF-IDF (Term Frequency–Inverse Document Frequency) was used to down-weight common terms and emphasize more topic-specific words. Unigrams and bigrams were included to capture short mathematical phrases.

Both approaches used a capped vocabulary size to control dimensionality.

---

## 5. Model Selection

Logistic Regression was chosen as the classification model due to:
- its simplicity and interpretability
- strong performance on high-dimensional sparse text data
- fast training and evaluation

No deep learning models were used for the primary classifier, in line with the goal of building a classical ML solution.

---

## 6. Ablation Study

An ablation study was conducted to evaluate the impact of the feature representation while keeping all other components constant.

The following settings were fixed:
- same training and testing splits
- same classifier (Logistic Regression)
- same hyperparameters

Only the vectorization method was varied.

### Results:
- **TF-IDF** achieved an accuracy of **72.06%**
- **Bag of Words** achieved an accuracy of **70.92%**

TF-IDF consistently outperformed Bag of Words, indicating that term weighting helps improve discrimination between math subtopics, especially for overlapping domains such as Algebra and Prealgebra.

This comparison serves as the primary ablation study for the project.

---

## 7. Evaluation Metrics

Model performance was evaluated using:
- overall accuracy
- class-wise precision, recall, and F1-score

The results showed higher performance for topics with more distinctive vocabulary (e.g., Precalculus) and lower performance for closely related categories (e.g., Prealgebra vs Algebra), which aligns with expectations.

---

## 8. Experiment Tracking

Due to the small number of controlled experiments and the focus on direct comparison between vectorization techniques, external experiment tracking tools such as Weights & Biases were not extensively used.

Results were analyzed through reproducible script execution and logged evaluation metrics.

---

## 9. Bonus Task: LLM-Based Solution Generation

As an extension to the core classification task, a large language model with over 7 billion parameters was deployed locally on Google Colab using GPU acceleration and quantization techniques.

The model was prompted to behave like a high school teacher and generate step-by-step, student-friendly explanations for selected math problems.

This component was implemented separately from the classification pipeline due to hardware constraints and is presented as a bonus demonstration rather than a dependency of the main system.

---

## 10. Conclusion

This project demonstrates that classical machine learning techniques, when combined with appropriate text representations, remain effective for structured educational text classification tasks.

The additional LLM-based solution generation highlights how modern language models can complement traditional ML systems by enhancing interpretability and student engagement.

---

