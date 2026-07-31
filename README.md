♻️ CleanSort AI — Intelligent Waste Classification

A machine learning project that classifies waste images into six categories — cardboard, glass, metal, paper, plastic, trash — using classical ML models implemented from scratch (no scikit-learn classifiers, no deep learning).

Course: Machine Learning · BS(CS), Section C

Dataset

TrashNet Dataset — ~2,500 images across six classes.

Pipeline

EDA → data cleaning → preprocessing (resize, grayscale, denoise) → feature extraction (HOG + LBP + Color Histogram) → train/val/test split → class balancing → model training → evaluation → error analysis.

Models
K-Nearest Neighbors (distance-weighted)
Gaussian Naive Bayes
Logistic Regression (One-vs-Rest, gradient descent)
Ensemble (majority vote across all three)
Results
Model	Accuracy
KNN	63.76%
Gaussian Naive Bayes	56.18%
Logistic Regression	64.89%
Ensemble	67.98%


How to Run:

Open the notebook in Google Colab
Run all cells top to bottom
Upload the TrashNet dataset when prompted
Use the built-in widget to upload a photo and classify it

Team:
Name	Role
Ehtasham Maqbool	Project Lead
Ahmad Khan	Data Collection & Preprocessing
M. Bilal	Evaluation Metrics & Testing
M. Saqib	UI/Deployment & Documentation Lead
