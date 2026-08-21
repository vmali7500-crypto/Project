from flask import Flask, request, render_template, redirect, url_for
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


def load_data():
    df = pd.read_csv("cleanedcar.csv")
    df.columns = df.columns.str.strip()
    df['company'] = df['company'].str.strip().str.title()
    df['name'] = df['name'].str.strip()

    companies = sorted(df['company'].unique())

    company_car = {}
    for company in companies:
        cars = df[df['company'] == company]['name'].unique()
        company_car[company] = list(cars)

    return companies, company_car


@app.route("/")
def home():
    companies, company_car = load_data()
    return render_template(
        "index.html",
        companies=companies,
        company_car=company_car
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    companies, company_car = load_data()

    if request.method == "POST":
        name = request.form["name"]
        company = request.form["company"]
        year = int(request.form["year"])
        kms_driven = int(request.form["kms_driven"])
        fuel_type = request.form["fuel_type"]

        if fuel_type == "":
            return render_template(
                "index.html",
                prediction_text="Please select fuel type",
                companies=companies,
                company_car=company_car
            )

        input_data = pd.DataFrame([{
            'name': name,
            'company': company,
            'year': year,
            'kms_driven': kms_driven,
            'fuel_type': fuel_type
        }])

        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated Price: Rs {round(prediction, 2)}",
            companies=companies,
            company_car=company_car
        )

    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)