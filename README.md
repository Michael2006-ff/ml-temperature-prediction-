# 🌡️ Temperature Prediction ML Project

An end-to-end Machine Learning project that predicts **today's temperature based on yesterday's temperature** and seasonal lag features. Built with Python, Scikit-Learn, Pandas, and Matplotlib.

---

## 📌 Features

- **Synthetic Weather Dataset**: 3 years of daily temperature data modeling seasonal harmonics and weather persistence autocorrelation.
- **Exploratory Data Analysis (EDA)**: Interactive data visualizations, temporal trend charts, scatter plots, and correlation heatmaps.
- **Multi-Model Comparison**: Evaluates 5 machine learning models side-by-side:
  - Polynomial Regression (Degree 2) 🏆
  - Linear Regression
  - Neural Network (Multi-Layer Perceptron)
  - Support Vector Regressor (SVR)
  - Random Forest Regressor
- **Web App UI**: Built-in interactive Web UI to test predictions right in your browser.
- **Jupyter Notebook**: Fully documented notebook (`temperature_prediction.ipynb`) for academic submission / college project presentations.

---

## 📈 Model Performance & Results

Models were evaluated on a 20% unseen test split:

| Model | RMSE (°C) ↓ | MAE (°C) ↓ | R² Score ↑ |
|---|---|---|---|
| **Polynomial Regression (Degree 2)** 🏆 | **1.733** | **1.381** | **0.9694** |
| **Linear Regression** | 1.761 | 1.397 | 0.9684 |
| **Neural Network (MLP)** | 1.847 | 1.481 | 0.9653 |
| **Support Vector Regressor (SVR)** | 1.981 | 1.570 | 0.9600 |
| **Random Forest Regressor** | 2.077 | 1.681 | 0.9560 |

---

## 📂 Project Structure

```
mlmodel/
├── data/
│   └── temperature_data.csv        # Generated dataset
├── models/
│   ├── best_model.pkl              # Saved trained model
│   └── scaler.pkl                  # Saved StandardScaler
├── outputs/                        # Saved EDA & evaluation plots
│   ├── eda_timeseries.png
│   ├── eda_scatter.png
│   ├── eda_correlation.png
│   ├── model_rmse_comparison.png
│   └── actual_vs_predicted.png
├── 01_generate_data.py             # Data generation script
├── 02_eda.py                       # Exploratory Data Analysis
├── 03_train_models.py              # Model training & comparison
├── 04_predict.py                   # Interactive CLI prediction
├── app.py                          # Web App UI server
├── temperature_prediction.ipynb    # Jupyter Notebook
├── requirements.txt                # Project dependencies
└── README.md                       # Documentation
```

---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Michael2006-ff/ml-temperature-prediction-.git
cd ml-temperature-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
# Generate synthetic dataset
python 01_generate_data.py

# Perform Exploratory Data Analysis
python 02_eda.py

# Train & evaluate models
python 03_train_models.py

# Make CLI prediction (e.g. 32.5 °C yesterday)
python 04_predict.py 32.5

# Run local Web UI
python app.py
```

---

## 🎓 License
Distributed under the MIT License. Built for educational and academic project purposes.
