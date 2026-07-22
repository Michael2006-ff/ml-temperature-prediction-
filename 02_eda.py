import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(csv_path='data/temperature_data.csv', output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print("=========================================")
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=========================================\n")
    print(f"Dataset Shape: {df.shape}")
    print("\n--- Summary Statistics ---")
    print(df[['temp_today', 'temp_yesterday', 'temp_7day_avg']].describe().round(2))
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # Set plot aesthetic
    sns.set_theme(style="whitegrid")
    
    # 1. Time Series Plot
    plt.figure(figsize=(14, 5))
    plt.plot(df['date'], df['temp_today'], label="Today's Temperature (deg C)", color='#1f77b4', alpha=0.8, linewidth=1.2)
    plt.plot(df['date'], df['temp_7day_avg'], label="7-Day Moving Avg (deg C)", color='#ff7f0e', linewidth=2.0)
    plt.title("Daily Temperature Time Series (3-Year Trend)", pad=15)
    plt.xlabel("Date")
    plt.ylabel("Temperature (deg C)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_timeseries.png'), dpi=300)
    plt.close()
    
    # 2. Scatter Plot: Yesterday vs Today
    plt.figure(figsize=(7, 6))
    sns.regplot(x='temp_yesterday', y='temp_today', data=df,
                scatter_kws={'alpha':0.4, 'color':'#2ca02c'},
                line_kws={'color':'red', 'linewidth':2})
    plt.title("Scatter Plot: Yesterday's vs Today's Temperature", pad=15)
    plt.xlabel("Yesterday's Temperature (deg C)")
    plt.ylabel("Today's Temperature (deg C)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_scatter.png'), dpi=300)
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(7, 5))
    numeric_df = df.drop(columns=['date'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_correlation.png'), dpi=300)
    plt.close()
    
    print(f"\n[SUCCESS] EDA Complete! All plots saved to '{output_dir}/' directory.")

if __name__ == '__main__':
    run_eda()
