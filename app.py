import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'Cleanedcar.csv')
model_PATH = os.path.join(BASE_DIR, 'model.pkl')

df = pd.read_csv(CSV_PATH)
model = pickle.load(open(model_PATH, 'rb'))

@app.route('/')
def home():
    companies = sorted(df['company'].unique())
    car_models = sorted(df['name'].unique())
    years = sorted(df['year'].unique(), reverse=True)
    fuel_types = df['fuel_type'].unique()
    
    company_car = {}
    for company in companies:
        company_car[company] = sorted(df[df['company'] == company]['name'].unique().tolist())
        
    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types,
        company_car=company_car
    )

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kilo_driven = int(request.form.get('kilo_driven'))

        prediction = model.predict(pd.DataFrame([[car_model, company, year, kilo_driven, fuel_type]], 
                                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
        
        return str(round(prediction[0], 2))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
