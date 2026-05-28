# Credit Card Fraud Detection

An **unsupervised machine learning** project that detects fraudulent credit card transactions using anomaly detection algorithms — without needing labeled training data for the fraud class.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZbXe4fNmpjRvZsrM-EFOdDOa3edHcR-T?usp=drive_link)

> **Educational Purpose** — Built as a cybersecurity/ML learning task to understand anomaly-based fraud detection, a technique widely used in real-world financial security systems.

---

## Problem Statement

Credit card fraud is a severe financial threat. Traditional supervised models require large amounts of labeled fraud data, which is hard to collect and imbalanced. This project uses **unsupervised anomaly detection** — treating fraud as an outlier in transaction patterns — which is closer to how real fraud detection systems work.

**Key challenge:** Only ~0.17% of transactions are fraudulent (highly imbalanced dataset).

---

## Algorithms Used

Three unsupervised outlier detection methods are compared:

| Algorithm | How It Works | Strength |
|-----------|-------------|----------|
| **Isolation Forest** | Randomly partitions data; anomalies are isolated faster | Fast, scalable, good with high-dimensional data |
| **Local Outlier Factor (LOF)** | Compares local density of a point to its neighbors | Good at detecting local anomalies in clusters |
| **One-Class SVM** | Learns a boundary around normal data; flags outside points | Powerful for non-linear decision boundaries |

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Feature | Details |
|---------|---------|
| Transactions | 284,807 |
| Fraudulent | 492 (~0.17%) |
| Features | 30 (V1–V28 PCA-transformed + Time + Amount) |
| Target | `Class` (0 = Normal, 1 = Fraud) |

> **Note:** Features V1–V28 are PCA-transformed for confidentiality. Only `Time` and `Amount` are original features.

---

## Project Workflow

```
Load Dataset (creditcard.csv)
        │
        ▼
Exploratory Data Analysis (EDA)
  ├── Class distribution (bar chart)
  ├── Amount per class (histogram)
  ├── Time vs Amount scatter (fraud timing)
  └── Correlation heatmap
        │
        ▼
Data Preparation
  ├── Sample 10% of data (random_state=1)
  ├── Calculate outlier_fraction = len(Fraud) / len(Valid)
  └── Split into X (features) and Y (target)
        │
        ▼
Train 3 Anomaly Detection Models
  ├── Isolation Forest
  ├── Local Outlier Factor
  └── One-Class SVM
        │
        ▼
Evaluate Each Model
  ├── Accuracy Score
  └── Classification Report (Precision, Recall, F1)
```

---

## Key EDA Insights

- **Class Imbalance:** Normal transactions vastly outnumber fraud — naive classifiers would just predict "Normal" and still get ~99.8% accuracy (which is meaningless).
- **Amount Distribution:** Fraudulent transactions tend to be smaller amounts, possibly to avoid detection.
- **Time Distribution:** Fraud occurs at all times but shows slight patterns at certain hours.
- **Correlation Heatmap:** PCA features (V1–V28) show low inter-correlation by design; `Amount` has the most variation.

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Click the **Open in Colab** badge at the top
2. Go to **Runtime → Run All**
3. When prompted, upload your `kaggle.json` API key

### Option 2: Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Download the dataset
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip

# Run the notebook
jupyter notebook credit_card_fraud_detection.ipynb
```

---

## Project Structure

```
credit-card-fraud-detection/
│
├── credit_card_fraud_detection.ipynb   # Main Colab notebook
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Ignore dataset & checkpoints
└── README.md                           # This file
```

---

## Dependencies

```
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
```

See `requirements.txt` for pinned versions.

---

## Cybersecurity Concepts Covered

| Concept | How It Applies Here |
|---------|-------------------|
| **Anomaly Detection** | Core technique — fraud treated as statistical outlier |
| **Unsupervised Learning** | No fraud labels needed during training |
| **Class Imbalance** | Real-world security datasets are always imbalanced |
| **Behavioral Profiling** | Normal patterns learned; deviations flagged |
| **Financial Security** | Direct application in banking and payment systems |
| **PCA / Feature Anonymization** | Real datasets anonymize sensitive features |

---

## Learning Resources

- [Kaggle Dataset — Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Scikit-learn: Anomaly Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Isolation Forest Paper (Liu et al., 2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [SMOTE for Imbalanced Learning](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

## License

Released for educational and research purposes only.

---

*Part of a cybersecurity & machine learning learning portfolio — exploring how ML powers real-world financial security systems.*
