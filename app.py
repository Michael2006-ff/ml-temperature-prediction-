import os
import json
import joblib
import pandas as pd
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

PORT = 5000

# Load Model & Scaler
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/best_model.pkl')

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Temperature Predictor AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            width: 100%;
            max-width: 480px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        h1 {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p.subtitle { color: #94a3b8; font-size: 14px; margin-bottom: 28px; }
        .form-group { margin-bottom: 20px; text-align: left; }
        label { display: block; font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #cbd5e1; }
        input[type="number"], input[type="date"] {
            width: 100%;
            padding: 14px 18px;
            border-radius: 12px;
            border: 1px solid #334155;
            background: #0f172a;
            color: #fff;
            font-size: 16px;
            outline: none;
            transition: all 0.2s;
        }
        input:focus { border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2); }
        button {
            width: 100%;
            padding: 16px;
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
            color: white;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.1s, opacity 0.2s;
            margin-top: 10px;
        }
        button:hover { opacity: 0.95; transform: translateY(-1px); }
        button:active { transform: translateY(1px); }
        .result-box {
            margin-top: 28px;
            padding: 20px;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid #334155;
            display: none;
            animation: fadeIn 0.3s ease-in-out;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .temp-val { font-size: 42px; font-weight: 700; color: #fbbf24; margin: 8px 0; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; background: #334155; color: #38bdf8; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Temp Predictor AI</h1>
        <p class="subtitle">Predict today's temperature using Polynomial Regression</p>
        
        <form id="predForm">
            <div class="form-group">
                <label>Yesterday's Temperature (deg C)</label>
                <input type="number" step="0.1" id="tempIn" value="28.5" required>
            </div>
            <div class="form-group">
                <label>Target Date</label>
                <input type="date" id="dateIn" required>
            </div>
            <button type="submit">Predict Today's Temp</button>
        </form>
        
        <div class="result-box" id="resultBox">
            <span class="badge">AI PREDICTION</span>
            <div class="temp-val" id="resTemp">-- deg C</div>
            <p id="resDate" style="color:#94a3b8; font-size:13px;"></p>
        </div>
    </div>

    <script>
        document.getElementById('dateIn').valueAsDate = new Date();
        
        document.getElementById('predForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const temp = document.getElementById('tempIn').value;
            const date = document.getElementById('dateIn').value;
            
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ temp: parseFloat(temp), date: date })
            });
            const data = await res.json();
            
            document.getElementById('resTemp').innerText = data.prediction + ' deg C';
            document.getElementById('resDate').innerText = 'Predicted for ' + data.date + ' | Model R2: 0.9694';
            document.getElementById('resultBox').style.display = 'block';
        });
    </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode('utf-8'))
        
    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))
            
            temp_yesterday = float(body.get('temp', 25.0))
            date_str = body.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            day_of_year = dt.timetuple().tm_yday
            month = dt.month
            
            input_df = pd.DataFrame([{
                'temp_yesterday': temp_yesterday,
                'day_of_year': day_of_year,
                'month': month,
                'temp_7day_avg': temp_yesterday
            }])
            
            model_name = type(model).__name__
            if model_name in ['SVR', 'MLPRegressor']:
                input_scaled = scaler.transform(input_df)
                pred = model.predict(input_scaled)[0]
            else:
                pred = model.predict(input_df)[0]
                
            response_data = {
                'prediction': round(float(pred), 2),
                'date': date_str
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

if __name__ == '__main__':
    server = HTTPServer(('localhost', PORT), RequestHandler)
    print("=========================================")
    print(f"[LIVE] WEB APP LIVE AT: http://localhost:{PORT}")
    print("Open this URL in your web browser!")
    print("=========================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
