# ACADEMIC PROJECT REPORT

## TOPIC: DAILY TEMPERATURE PREDICTION USING MACHINE LEARNING

---

### **Project Metadata**
- **Project Title**: Temperature Prediction System based on Historical Lag Features
- **Target Application**: Short-term Weather Forecasting & Regression Analysis
- **GitHub Repository**: [HaarisAbdullah/ml-temperature-prediction](https://github.com/HaarisAbdullah/ml-temperature-prediction)
- **Primary Model**: Polynomial Regression (Degree 2)
- **Evaluation Metric**: Root Mean Squared Error (RMSE) & Coefficient of Determination ($R^2$)

---

## ABSTRACT

Short-term weather and temperature forecasting plays a vital role in agriculture, energy demand estimation, and daily human activity planning. Traditional meteorological models rely on complex physical fluid dynamics equations requiring supercomputing power. In contrast, Machine Learning (ML) techniques can effectively capture temporal patterns and non-linear seasonal variations directly from historical observational data.

This project presents a complete machine learning system designed to predict today's temperature based on yesterday's temperature, annual seasonality markers, and short-term rolling averages. A 3-year daily weather dataset (1,088 samples) modeling seasonal harmonics and autocorrelated noise was processed. Five distinct regression algorithms—Linear Regression, Polynomial Regression (Degree 2), Support Vector Regression (SVR), Random Forest Regressor, and a Multi-Layer Perceptron (MLP) Neural Network—were trained and benchmarked. Polynomial Regression achieved the highest predictive performance with an $R^2$ score of **0.9694** and a Root Mean Squared Error (RMSE) of **1.733 °C**. An interactive Web application and CLI interface were deployed to enable real-time predictions.

---

## 1. INTRODUCTION & PROBLEM STATEMENT

### 1.1 Background
Temperature forecasting is fundamentally a time-series regression problem. Because the atmosphere exhibits thermal inertia, tomorrow's temperature is strongly correlated with today's temperature. However, linear extrapolation alone is insufficient due to annual seasonal shifts and localized weather variations.

### 1.2 Objective
The core objectives of this project are:
1. To formulate a lag-based predictive feature set using historical daily temperatures.
2. To conduct Exploratory Data Analysis (EDA) to understand seasonal trends and feature correlations.
3. To train, evaluate, and compare 5 regression machine learning models on unseen test data.
4. To export the trained model and build a lightweight inference web interface for user interaction.

---

## 2. DATASET & FEATURE ENGINEERING

### 2.1 Data Synthesizing & Modeling
To model real-world meteorological dynamics, a synthetic dataset spanning 3 years (1,095 days) was constructed using the following mathematical formulation:

1. **Annual Seasonal Trend Component (Sinusoidal Harmonic)**:
   $$\text{Base Temp}(t) = 25.0 + 12.0 \cdot \sin\left(\frac{2\pi (d - 80)}{365.25}\right)$$
   where $d$ represents the day of the year (1 to 365).

2. **Thermal Inertia & Autocorrelation (AR-1 Noise Process)**:
   $$\epsilon_t = 0.75 \cdot \epsilon_{t-1} + \mathcal{N}(0, 1.8)$$
   where $\epsilon_t$ models weather persistence across consecutive days.

3. **Final Temperature Formulation**:
   $$\text{temp\_today}_t = \text{Base Temp}(t) + \epsilon_t$$

### 2.2 Feature Extraction
The extracted input feature vector $\mathbf{X}$ comprises:
- `temp_yesterday`: $T_{t-1}$ (Primary Lag-1 feature)
- `day_of_year`: $d \in [1, 365]$ (Captures annual solar elevation cycle)
- `month`: $m \in [1, 12]$ (Coarse seasonal indicator)
- `temp_7day_avg`: $\frac{1}{7}\sum_{i=1}^{7} T_{t-i}$ (Moving weekly baseline)

Target Variable $y$: `temp_today` ($T_t$)

---

## 3. EXPLORATORY DATA ANALYSIS (EDA)

EDA was performed to validate data consistency and underlying mathematical assumptions prior to model training:

1. **Time-Series Trend Plot**: Confirmed periodic annual oscillations bounded between 7.80 °C and 44.07 °C with a mean temperature of 25.30 °C.
2. **Scatter Correlation Plot**: Demonstrated a strong positive linear-polynomial relationship between `temp_yesterday` and `temp_today`.
3. **Correlation Heatmap**: Showed high Pearson correlation coefficients ($r > 0.95$) between lagged temperatures and current temperature.

---

## 4. METHODOLOGY & MODEL ARCHITECTURE

The dataset (1,088 valid samples) was split sequentially into **80% Training Data (870 samples)** and **20% Testing Data (218 samples)**. Features were standardized using `StandardScaler` for scale-sensitive algorithms.

### Evaluated Algorithms:
1. **Linear Regression**: Baseline parametric model fitting an optimal hyper-plane.
2. **Polynomial Regression (Degree 2)**: Expands features into second-degree combinations ($T_{t-1}^2, d^2, T_{t-1} \cdot d$) to capture seasonal curvature.
3. **Support Vector Regressor (SVR)**: Uses Radial Basis Function (RBF) kernel with hyper-parameters $C=10.0, \epsilon=0.1$.
4. **Random Forest Regressor**: Non-parametric ensemble consisting of 100 decision trees.
5. **Multi-Layer Perceptron (MLP Neural Network)**: Deep neural architecture with hidden layers $(64, 32)$ and ReLU activations.

---

## 5. EXPERIMENTAL RESULTS & COMPARISON

Models were evaluated using three standard regression metrics on the unseen 20% test partition:
- **Root Mean Squared Error (RMSE)**: Penalizes larger prediction errors.
- **Mean Absolute Error (MAE)**: Measures average absolute deviation.
- **Coefficient of Determination ($R^2$)**: Represents variance explained by the model.

### Metric Comparison Table:

| Model | RMSE (°C) ↓ | MAE (°C) ↓ | $R^2$ Score ↑ |
|---|---|---|---|
| **Polynomial Regression (Degree 2)** 🏆 | **1.733** | **1.381** | **0.9694** |
| **Linear Regression** | 1.761 | 1.397 | 0.9684 |
| **Neural Network (MLP)** | 1.847 | 1.481 | 0.9653 |
| **Support Vector Regressor (SVR)** | 1.981 | 1.570 | 0.9600 |
| **Random Forest Regressor** | 2.077 | 1.681 | 0.9560 |

### Key Findings:
- **Polynomial Regression** achieved the best overall performance ($R^2 = 0.9694$), correctly capturing the non-linear curvature caused by seasonal transitions.
- **Linear Regression** performed competitively ($R^2 = 0.9684$), proving that lag-1 temperature is a dominant predictor.
- **Tree-based ensembles (Random Forest)** suffered slight degradation ($R^2 = 0.9560$) due to step-wise boundary decisions on continuous smooth time series.

---

## 6. SYSTEM IMPLEMENTATION & DEPLOYMENT

The project software architecture is modularized into dedicated Python modules:
- `01_generate_data.py`: Data generation & CSV serialization.
- `02_eda.py`: Automated plot rendering and statistics export.
- `03_train_models.py`: Model pipeline execution & `joblib` model serialization (`best_model.pkl`).
- `04_predict.py`: Command-line inference module.
- `app.py`: Web dashboard providing an interactive browser UI (`http://localhost:5000`).

---

## 7. CONCLUSION & FUTURE SCOPE

### 7.1 Conclusion
This project successfully demonstrates that daily temperature can be predicted with high accuracy ($R^2 > 0.96$, $\text{RMSE} < 1.75\text{ °C}$) using yesterday's temperature and calendar lag features. Polynomial Regression proved to be the most optimal model for this task.

### 7.2 Future Scope
1. **Multi-Step Ahead Forecasting**: Extending predictions to 7-day and 14-day horizons using Recurrent Neural Networks (LSTM/GRU).
2. **Exogenous Feature Integration**: Incorporating atmospheric pressure, relative humidity, wind speed, and cloud cover data from real weather APIs (e.g., OpenWeatherMap).

---

## 8. REFERENCES
1. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
