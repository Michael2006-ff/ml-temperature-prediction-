import os
import sys
import joblib
import pandas as pd
from datetime import datetime

def predict_today_temp(temp_yesterday, date_str=None, models_dir='models'):
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, 'best_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please run '03_train_models.py' first!")
        
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    
    if date_str is None:
        target_date = datetime.now()
    else:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        
    day_of_year = target_date.timetuple().tm_yday
    month = target_date.month
    
    temp_7day_avg = temp_yesterday
    
    input_data = pd.DataFrame([{
        'temp_yesterday': temp_yesterday,
        'day_of_year': day_of_year,
        'month': month,
        'temp_7day_avg': temp_7day_avg
    }])
    
    model_name = type(model).__name__
    if model_name in ['SVR', 'MLPRegressor']:
        input_scaled = scaler.transform(input_data)
        predicted_temp = model.predict(input_scaled)[0]
    else:
        predicted_temp = model.predict(input_data)[0]
        
    return round(predicted_temp, 2), target_date.strftime('%Y-%m-%d')

if __name__ == '__main__':
    print("=========================================")
    print("TEMPERATURE PREDICTION SYSTEM")
    print("=========================================\n")
    
    try:
        if len(sys.argv) > 1:
            temp_in = float(sys.argv[1])
            date_in = sys.argv[2] if len(sys.argv) > 2 else None
        else:
            temp_in = float(input("Enter Yesterday's Temperature (deg C) [e.g., 28.5]: ") or "28.5")
            date_in = input("Enter Today's Date (YYYY-MM-DD) [Press Enter for today]: ") or None
        
        predicted, date_used = predict_today_temp(temp_in, date_in)
        
        print("\n-----------------------------------------")
        print(f"Target Date:               {date_used}")
        print(f"Yesterday's Temperature:   {temp_in} deg C")
        print(f"PREDICTED TODAY'S TEMP:    {predicted} deg C")
        print("-----------------------------------------\n")
    except Exception as e:
        print(f"[ERROR] Error making prediction: {e}")
