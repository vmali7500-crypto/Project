import os
import pickle
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Absolute paths relative to app.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'Cleanedcar.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

# Load model and dataset safely
model = pickle.load(open(MODEL_PATH, "rb"))
df = pd.read_csv(CSV_PATH)

def get_dropdown_data():
    companies = sorted(df['company'].unique())
    company_car = {}
    car_fuel = {}

    for company in companies:
        company_car[company] = sorted(df[df['company'] == company]['name'].unique().tolist())
    
    # Map car model names to their available fuel types
    for car_name in df['name'].unique():
        car_fuel[car_name] = sorted(df[df['name'] == car_name]['fuel_type'].dropna().unique().tolist())

    return companies, company_car, car_fuel

@app.route("/")
def home():
    companies, company_car, car_fuel = get_dropdown_data()
    return render_template("index.html", companies=companies, company_car=company_car, car_fuel=car_fuel)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Redirect GET requests (e.g., manually opening /predict in address bar) back to home
    if request.method == "GET":
        return redirect(url_for("home"))

    companies, company_car, car_fuel = get_dropdown_data()
    
    name = request.form.get("name")
    company = request.form.get("company")
    year = int(request.form.get("year"))
    kms_driven = int(request.form.get("kms_driven"))
    fuel_type = request.form.get("fuel_type")

    if not fuel_type:
        return render_template("index.html", 
                               companies=companies, 
                               company_car=company_car, 
                               car_fuel=car_fuel,
                               prediction_text="Please select fuel type")

    input_data = pd.DataFrame([{
        'name': name,
        'company': company,
        'year': year,
        'kms_driven': kms_driven,
        'fuel_type': fuel_type
    }])

    prediction = model.predict(input_data)[0]

    return render_template("index.html", 
                           companies=companies, 
                           company_car=company_car, 
                           car_fuel=car_fuel,
                           prediction_text=f"Estimated Price: ₹ {round(prediction, 2)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
