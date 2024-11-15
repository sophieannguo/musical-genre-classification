# **Audio-Based Genre Classification Using Machine Learning: A Feature-Driven Approach**

This project explores music genre classification using the GTZAN dataset, combining feature engineering, dimensionality reduction, and various machine learning techniques. Key approaches include feature analysis, PCA, LDA, QDA, Neural Networks, and K-Nearest Neighbors, achieving high classification accuracy.

---

## **Overview**

- **Dataset**: GTZAN Dataset, segmented into 3-second audio clips from 10 genres (1,000 clips per genre).
- **Features**: Extracted data includes 57 attributes such as:
  - Chroma STFT, Spectral Centroid, MFCCs (1-20), Tempo, RMS, Harmony, etc.
- **Preprocessing**:
  - Dimensionality reduction with PCA.
  - Standardization and multicollinearity analysis using LASSO regression.

---

## **Machine Learning Models**

- **Linear Discriminant Analysis (LDA)**:
  - Achieved 67.22% accuracy.
- **Quadratic Discriminant Analysis (QDA)**:
  - Achieved 76.05% accuracy.
- **Feedforward Neural Networks (FNN)**:
  - Test accuracy of 84–87% after hyperparameter optimization.
- **Convolutional Neural Networks (CNN)**:
  - Trained on Mel Spectrograms, achieving 76.04% accuracy.
- **K-Nearest Neighbors (KNN)**:
  - Best-performing model with an accuracy of 92.22% using LOOCV.

---

## **Setup and Installation**

### **Prerequisites**
- Python 3.8 or later
- R and RMarkdown (for integrated statistical and visual analysis)
- Required libraries:
  - Python: `tensorflow`, `keras`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
  - R: `dplyr`, `ggplot2`, `caret`, `glmnet`, `heatmaply`, `plotly`, `tidyverse`

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audio-genre-classification.git
   cd audio-genre-classification
