# -Intelligent-Waste-Classification-using-Machine-Learning-


A multi-class image classification system that sorts waste photos into six categories using classical machine learning models implemented entirely from scratch — no scikit-learn classifiers, no deep learning.

Course: Machine Learning · BS(CS), Section C

Table of Contents
Overview
Results
Pipeline
Models
Error Analysis
Project Structure
Getting Started
Deployment
Team
Limitations
AI Usage Disclosure
References
Overview

Manual waste sorting is slow, costly, and error-prone. This project builds an automated classifier that sorts images into cardboard, glass, metal, paper, plastic, and trash, using the TrashNet dataset (~2,500 images).

Every model — KNN, Gaussian Naive Bayes, and Logistic Regression — along with every evaluation metric, is implemented from first principles in NumPy. No sklearn.linear_model, no sklearn.neighbors, no sklearn.metrics.

Results
Model	Accuracy	Macro Precision	Macro Recall	Macro F1
Majority-class baseline	23.31%	—	—	—
Random (weighted) baseline	17.70%	—	—	—
KNN (distance-weighted)	63.76%	64.83%	63.71%	63.76%
Gaussian Naive Bayes	56.18%	55.30%	58.68%	55.98%
Logistic Regression (OvR)	64.89%	64.97%	64.97%	64.92%
Ensemble (majority vote)	67.98%	71.1%	68.2%	68.1%

All three models comfortably beat both baselines, confirming genuine learning from the feature set — the gap to the proposal's 80–85% target is a feature-representation limitation, discussed below, not a failure to learn.

Pipeline
EDA — class distribution, sample images, dimension analysis, RGB/pixel-intensity histograms
Data cleaning — removes corrupt images, exact duplicates (MD5 hash), near-duplicates (perceptual hash), and blur outliers (2,527 → 2,366 images)
Preprocessing — resize (128×128), grayscale, Gaussian blur, normalization
Feature extraction — HOG (shape) + LBP (texture) + Color Histogram (color) → 1,886-dimensional feature vector
Split — manual 70/15/15 train/val/test (NumPy permutation, no sklearn)
Class balancing — random oversampling of minority classes, training set only
Baseline models — majority-class and random, as a reference floor
Three from-scratch models — see below
Ensemble — majority vote across all three
Evaluation — accuracy, precision, recall, F1, confusion matrices, all computed manually
Error analysis — misclassified image inspection, confused-class-pair analysis
Prototype interface — in-notebook upload-and-classify widget
Models
Model	Key implementation detail
K-Nearest Neighbors	Euclidean distance, distance-weighted voting, K tuned on validation set
Gaussian Naive Bayes	Log-posterior probabilities, per-class mean/variance estimation
Logistic Regression	One-vs-Rest, batch gradient descent, L2 regularization, learning rate × λ grid search
Error Analysis

Of 356 test images, the best single model (Logistic Regression) correctly classified 231 (64.9%).

Most confused class pairs:

True Class	Predicted As	Occurrences
Glass	Plastic	18
Metal	Glass	11
Plastic	Glass	10

Glass and plastic have the two highest per-class error rates (44.6% and 47.7%). This confirms our proposal's Risk Assessment: HOG/LBP/color-histogram features capture shape, texture, and color, but not material properties like transparency or reflectivity — exactly what distinguishes a glass jar from a plastic one to a human eye.

Project Structure
.
├── Final_Project_Waste_Classification_ML_Completed.ipynb   # main notebook — run this
├── README.md
│
├── huggingface-space/          # optional Gradio deployment
│   ├── app.py
│   ├── requirements.txt
│   └── cleansort_ai_model.pkl
│
└── static-demo/                # optional client-side demo, runs in-browser
    ├── index.html
    ├── style.css
    ├── app.js
    └── model.json
Getting Started

This project runs in Google Colab.

Open Final_Project_Waste_Classification_ML_Completed.ipynb in Colab
Run all cells top to bottom, in order
When prompted (Cell 1.3), upload the TrashNet dataset as a .zip or .rar — download it from Kaggle
Use the widget in Cell 10.3 to upload your own photo and get a live prediction

Feature extraction (Part 4) is the slowest step — typically 3–8 minutes.

Deployment (optional)

The trained model can also be used outside the notebook:

Hugging Face Space — a Gradio web app with an auto-generated API (huggingface-space/)
Static demo — a plain HTML/CSS/JS page that runs the full pipeline client-side, no server required (static-demo/)

These are supplementary demonstrations and not part of the graded notebook deliverable.

Team
Name	Role
Ehtasham Maqbool	Project Lead
Ahmad Khan	Data Collection & Preprocessing
M. Bilal	Evaluation Metrics & Testing
M. Saqib	UI/Deployment & Documentation Lead
Limitations
Handcrafted features cap achievable accuracy below what a CNN/transfer-learning approach (e.g. MobileNetV2) could reach — a deliberate scope decision per the approved proposal, not an oversight
Class-balancing via oversampling helped KNN but slightly hurt Naive Bayes, since it equalizes training-set priors while the real class distribution stays imbalanced
KNN's "training" is simply memorizing the training set, so it scales poorly with dataset size
AI Usage Disclosure

Claude (Anthropic) was used to explain how these algorithms and metrics work internally; that understanding was then used to implement everything from scratch in NumPy rather than using library functions. It was also used for debugging, code review against our rubric, deployment support, and report drafting. Full disclosure is in the Final Report. No AI tool was used to fabricate results — all figures come from running the notebook on our actual dataset.

References
TrashNet Dataset, Kaggle — https://www.kaggle.com/datasets/feyzazkefe/trashnet
Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification.
Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution Gray-Scale and Rotation Invariant Texture Classification using Local Binary Patterns.
