import os
import numpy as np
import pandas as pd

def generate_temperature_data(num_days=1095, random_seed=42):
    np.random.seed(random_seed)
    
    dates = pd.date_range(start='2021-01-01', periods=num_days, freq='D')
    day_of_year = dates.dayofyear.values
    month = dates.month.values
    
    # Base temperature model: Sinusoidal trend representing annual season cycles
    # Base avg temp = 25°C, variation amplitude = 12°C
    base_temp = 25.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    
    # Adding autocorrelation (yesterday's noise influences today's noise)
    noise = np.zeros(num_days)
    current_noise = 0
    for t in range(num_days):
        current_noise = 0.75 * current_noise + np.random.normal(0, 1.8)
        noise[t] = current_noise
        
    temp_today = base_temp + noise
    
    df = pd.DataFrame({
        'date': dates,
        'temp_today': np.round(temp_today, 2),
        'day_of_year': day_of_year,
        'month': month
    })
    
    # Feature engineering: Lag features
    df['temp_yesterday'] = df['temp_today'].shift(1)
    df['temp_7day_avg'] = df['temp_today'].shift(1).rolling(window=7).mean()
    
    # Drop initial rows with NaNs resulting from shifts/rolling windows
    df = df.dropna().reset_index(drop=True)
    
    # Reorder columns
    cols = ['date', 'temp_today', 'temp_yesterday', 'day_of_year', 'month', 'temp_7day_avg']
    df = df[cols]
    
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_temperature_data()
    output_path = os.path.join('data', 'temperature_data.csv')
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Successfully generated synthetic dataset with {len(df)} records!")
    print(f"[SAVED] File path: {output_path}")
    print("\n--- Sample Data ---")
    print(df.head())
