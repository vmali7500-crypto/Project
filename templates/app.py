from flask import Flask, request, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
@app.route("/")
def home():
    return render_template("index.html")
import pandas as pd

@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    company = request.form["company"]
    year = int(request.form["year"])
    kms_driven = int(request.form["kms_driven"])
    fuel_type = request.form["fuel_type"]
    if fuel_type == "":
        return render_template("index.html", prediction_text="Please select fuel type")

    input_data = pd.DataFrame([{
        'name': name,
        'company': company,
        'year': year,
        'kms_driven': kms_driven,
        'fuel_type': fuel_type
    }])

    prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction_text=f"Estimated Price: Rs {round(prediction, 2)}")
if __name__ == "__main__":
    app.run(debug=True)