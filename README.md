# 🏛️ LegalTech NLP Classifier

**Detecting Abusive Clauses in Polish Consumer Contracts using NLP**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Models](#-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Explainability (XAI)](#-explainability-xai)
- [Deployment Recommendations](#-deployment-recommendations)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [References](#-references)
- [Author](#-author)

---

## 🎯 Overview

This project implements and compares two approaches for detecting **abusive clauses** (*klauzule abuzywne*) in Polish consumer contracts:

1. **Traditional ML**: TF-IDF vectorization + Logistic Regression (baseline)
2. **Deep Learning**: Fine-tuned HerBERT (Polish BERT model)

### What are Abusive Clauses?

Abusive clauses are contract provisions that:
- Violate consumer rights under **EU Directive 93/13/EEC** and Polish Civil Code
- Create significant imbalance between consumer and business obligations
- Are not individually negotiated with the consumer

Examples include: unilateral contract modification rights, excessive penalties, limitation of liability, and automatic contract renewal without consent.

The Polish Office of Competition and Consumer Protection (UOKiK) maintains a register of such clauses, making this classification task legally significant for contract compliance checking.

---

## ✨ Features

- 📊 **Comprehensive EDA** - Detailed exploratory data analysis with visualizations
- 🤖 **Dual Model Approach** - Compare traditional ML with deep learning
- ⚖️ **Class Imbalance Handling** - Weighted loss functions for imbalanced data
- 🔍 **Explainability (XAI)** - LIME-based explanations for model predictions
- 📈 **Threshold Analysis** - Optimize classification threshold for legal applications
- 💾 **Model Export** - Save and download trained models for deployment

---

## 📁 Project Structure

```
NLP_project/
├── legaltech-nlp-classifier.ipynb  # Main Jupyter notebook with full analysis
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── dataset/
│   ├── train.parquet               # Training data
│   ├── val.parquet                 # Validation data
│   └── test.parquet                # Test data
```

---

## 📊 Dataset

**Source**: [laugustyniak/abusive-clauses-pl](https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl) on HuggingFace

The dataset contains clauses from real Polish consumer contracts, manually annotated by legal experts.

| Split | Samples | Abusive (%) | Safe (%) |
|-------|---------|-------------|----------|
| Train | ~4,300  | ~45%        | ~55%     |
| Val   | ~1,500  | ~70%        | ~30%     |
| Test  | ~3,500  | ~32%        | ~68%     |

### Class Imbalance

This project addresses imbalance through:
- **Class-weighted loss functions**
- **Balanced accuracy and F1-score metrics**
- **Threshold optimization for high recall**

---

## 🤖 Models

### 1. Baseline: TF-IDF + Logistic Regression

- **Vectorization**: TF-IDF with 5,000 features
- **Classifier**: Logistic Regression with balanced class weights
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Pros**: Fast, interpretable, good baseline
- **Cons**: No contextual understanding

### 2. Deep Learning: HerBERT

- **Model**: [allegro/herbert-base-cased](https://huggingface.co/allegro/herbert-base-cased)
- **Architecture**: BERT-based transformer pretrained on Polish text
- **Training**: Fine-tuned for sequence classification with weighted loss
- **Pros**: Contextual understanding, state-of-the-art performance
- **Cons**: Requires GPU, longer training time

#### Why HerBERT?

| Model | Pros | Cons |
|-------|------|------|
| **HerBERT** ✓ | Native Polish tokenization, trained on diverse Polish text | Requires GPU |
| Multilingual BERT | Supports 100+ languages | Suboptimal Polish tokenization |
| Polish-RoBERTa | Strong performance | Larger model, more resources |
| XLM-RoBERTa | State-of-the-art multilingual | Overkill for monolingual task |

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for HerBERT training)
- Conda or pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Norbecek/legaltech-nlp-classifier.git
   cd legaltech-nlp-classifier
   ```

2. **Create a virtual environment**
   ```bash
   # Using conda
   conda create -n nlp_project python=3.10
   conda activate nlp_project

   # Or using venv
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
   # Additional dependencies for full functionality
   pip install transformers datasets evaluate sacremoses lime
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Set up HuggingFace authentication**
   ```bash
   huggingface-cli login
   ```

---

## 🚀 Usage

### Running the Notebook

1. Open `legaltech-nlp-classifier.ipynb` in Jupyter or VS Code
2. Ensure GPU runtime is enabled (for HerBERT training)
3. Run cells sequentially

### Quick Start with Pre-trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "allegro/herbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("./trained_model_herbert")

# Classify a clause
def classify_clause(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    pred = "ABUSIVE" if probs[0, 1] > 0.5 else "SAFE"
    confidence = probs[0, 1].item() if pred == "ABUSIVE" else probs[0, 0].item()
    return pred, confidence

# Example
text = "Sprzedawca zastrzega sobie prawo do jednostronnej zmiany cen."
prediction, confidence = classify_clause(text)
print(f"Prediction: {prediction} (confidence: {confidence:.2%})")
```

---

## 📈 Results

### Model Comparison

| Model | F1-Score | Balanced Accuracy | ROC-AUC |
|-------|----------|-------------------|---------|
| LR Baseline | ~0.70 | ~0.75 | ~0.85 |
| LR Tuned | ~0.72 | ~0.77 | ~0.87 |
| **HerBERT** | **~0.85** | **~0.88** | **~0.94** |

### Key Metrics Explained

- **F1-Score**: Harmonic mean of precision and recall - best for imbalanced data
- **Balanced Accuracy**: Average recall across classes - unaffected by class imbalance
- **ROC-AUC**: Model's ability to distinguish classes across all thresholds

---

## 🔍 Explainability (XAI)

Model predictions are explained using **LIME (Local Interpretable Model-agnostic Explanations)**:

1. Creates perturbations of input text by removing words
2. Gets model predictions for each perturbation
3. Fits interpretable model to learn word influence
4. Highlights words pushing prediction toward each class

### Sample Explanation

```
📋 TRUE POSITIVE
Clause: "Sprzedawca zastrzega sobie prawo do jednostronnej zmiany..."
Prediction: ABUSIVE (confidence: 94.2%)

Top influential words:
   zastrzega: +0.3421 → ABUSIVE
   jednostronnej: +0.2845 → ABUSIVE
   zmiany: +0.1923 → ABUSIVE
```

---

## 🚀 Deployment Recommendations

### Threshold Selection

| Use Case | Threshold | Trade-off |
|----------|-----------|-----------|
| **Consumer Protection** | 0.3-0.4 | High recall, more false positives |
| **Balanced Operation** | 0.5 | Default threshold |
| **High-throughput Screening** | Optimal F1 | Best overall performance |

### Why Lower Threshold for Legal Applications?

- **False negatives** (missed abusive clauses) have higher legal/financial consequences
- False positives can be filtered by human review
- A "flag for review" system with high recall is preferable

---

## ⚠️ Limitations

1. **Dataset scope**: Limited to Polish consumer contracts; may not generalize to other jurisdictions
2. **Annotation quality**: Model performance bounded by human annotator agreement
3. **Temporal drift**: Legal language evolves; periodic retraining recommended
4. **Context limitation**: 256-token limit may truncate longer clauses
5. **Language**: Currently supports only Polish text

---

## 🔮 Future Work

- [ ] Apply better preprocessing (stop word removal, normalization)
- [ ] Mask cities/surnames to prevent learning specific patterns
- [ ] Incorporate clause context (surrounding clauses in contracts)
- [ ] Multi-label classification for specific abusiveness categories
- [ ] Cross-lingual transfer to other Slavic languages
- [ ] Active learning for continuous model improvement
- [ ] Deploy as REST API or web application

---

## 📚 References

1. **Dataset**: Augustyniak, Ł., et al. (2020). *Abusive Clauses Detection in Polish Consumer Contracts*. [HuggingFace](https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl)

2. **HerBERT Model**: Mroczkowski, R., et al. (2021). *HerBERT: Efficiently Pretrained Transformer-based Language Model for Polish*. [HuggingFace](https://huggingface.co/allegro/herbert-base-cased)

3. **XAI Method**: Ribeiro, M. T., et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*. KDD 2016. (LIME)

4. **Legal Framework**: EU Directive 93/13/EEC on Unfair Terms in Consumer Contracts

5. **Class Imbalance**: He, H., & Garcia, E. A. (2009). *Learning from Imbalanced Data*. IEEE TKDE.

---

## 👤 Author

**norbek**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Allegro](https://allegro.tech/) for the HerBERT model
- [HuggingFace](https://huggingface.co/) for the datasets and transformers library
- Polish Office of Competition and Consumer Protection (UOKiK) for maintaining the abusive clauses register

---

<p align="center">
  <i>Built with ❤️ for consumer protection</i>
</p>
