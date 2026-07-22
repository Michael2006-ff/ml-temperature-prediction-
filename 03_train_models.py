import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_and_evaluate(csv_path='data/temperature_data.csv', models_dir='models', output_dir='outputs'):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # Feature Selection & Target Definition
    X = df[['temp_yesterday', 'day_of_year', 'month', 'temp_7day_avg']]
    y = df['temp_today']
    
    # Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Standard Scaler for models sensitive to scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models dictionary
    models = {
        'Linear Regression': LinearRegression(),
        'Polynomial Regression (Degree 2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'Support Vector Regressor (SVR)': SVR(kernel='rbf', C=10.0, epsilon=0.1),
        'Neural Network (MLP)': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    results = {}
    predictions = {}
    best_model_name = None
    best_r2 = -float('inf')
    best_model = None
    
    print("=========================================")
    print("MODEL TRAINING & EVALUATION")
    print("=========================================\n")
    
    for name, model in models.items():
        # Train model
        if name in ['Support Vector Regressor (SVR)', 'Neural Network (MLP)']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        # Metrics computation
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        predictions[name] = y_pred
        
        print(f"[*] Model: {name}")
        print(f"   - RMSE: {rmse:.3f} deg C")
        print(f"   - MAE:  {mae:.3f} deg C")
        print(f"   - R2:   {r2:.4f}\n")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model

    # Save metrics summary table
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
    
    # Save scaler and best model
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    
    print("=========================================")
    print(f"[BEST MODEL] {best_model_name} (R2 = {best_r2:.4f})")
    print(f"[SAVED] Best model saved to: '{models_dir}/best_model.pkl'")
    print("=========================================\n")
    
    # Visualization 1: Model Metrics Comparison Chart
    plt.figure(figsize=(10, 5))
    sns.barplot(x=results_df.index, y=results_df['RMSE'], hue=results_df.index, palette='Blues_d', legend=False)
    plt.title("Model Comparison - Root Mean Squared Error (RMSE)", pad=15)
    plt.ylabel("RMSE (deg C) [Lower is better]")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_rmse_comparison.png'), dpi=300)
    plt.close()
    
    # Visualization 2: Actual vs Predicted Time Series (Test Set)
    plt.figure(figsize=(14, 6))
    test_dates = pd.to_datetime(df.iloc[X_test.index]['date'])
    plt.plot(test_dates, y_test.values, label='Actual Temperature', color='black', alpha=0.7, linewidth=1.5)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (name, y_pred) in enumerate(predictions.items()):
        plt.plot(test_dates, y_pred, label=name, color=colors[idx % len(colors)], alpha=0.7, linestyle='--')
        
    plt.title("Actual vs Predicted Temperatures on Test Set", pad=15)
    plt.xlabel("Date")
    plt.ylabel("Temperature (deg C)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()
    
    return results_df

if __name__ == '__main__':
    train_and_evaluate()
