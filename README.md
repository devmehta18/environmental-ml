# 🌱 Machine Learning on Environmental Indicators

This project analyzes environmental data to model forest cover loss in the Brazilian Amazon using various machine learning algorithms. It was developed as part of my MSc at King’s College London and contributes to climate change research and conservation strategies.

---

## 🧠 Objective

To identify patterns in environmental indicators and use regression models to predict deforestation levels, supporting global monitoring and policy-making.

---

## 📊 Dataset

- 24 environmental indicators
- 15+ years of satellite data from Brazilian Amazon
- Data preprocessing & feature engineering for regression based modeling

---

## 🚀 Machine Learning Models Used

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting (Best: R² = **0.807**)
- Support Vector Machine
- Multi-Layer Perceptron (MLP)

---

## 📁 Project Structure

environmental-ml/
│
├── data/                   # (Optional) Small sample dataset or link in README
├── notebooks/              # Jupyter notebooks with EDA + modeling
├── src/                    # Scripts for preprocessing, modeling, etc.
│   ├── data_processing.py
│   ├── model_training.py
│   └── evaluation.py
├── models/                 # Saved models (.pkl or .joblib if applicable)
├── README.md
├── requirements.txt
└── .gitignore
