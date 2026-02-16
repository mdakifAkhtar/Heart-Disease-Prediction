from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Predict route triggered")

        # Get values from form
        sex = float(request.form['sex_m'])
        age = float(request.form['age'])
        cp_ata = float(request.form['cp_ata'])
        exercise = float(request.form['exercise'])
        st_up = float(request.form['st_up'])
        fasting = float(request.form['fasting'])
        maxhr = float(request.form['maxhr'])
        st_flat = float(request.form['st_flat'])
        oldpeak = float(request.form['oldpeak'])

        # Feature order must match training
        features = np.array([[sex, age, cp_ata, exercise,st_up, fasting, maxhr,st_flat, oldpeak]])

        # Scale numerical columns (Age, MaxHR, Oldpeak)
        numerical_values = features[:, [1, 6, 8]]
        scaled_values = scaler.transform(numerical_values)
        features[:, [1, 6, 8]] = scaled_values

        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        # Calculate risk %
        risk_percent = round(probability * 100, 2)

        # Result message
        if prediction == 1:
            result = f"Heart Disease Detected (Risk: {risk_percent}%)"
        else:
            result = f"No Heart Disease (Risk: {risk_percent}%)"

        # Send data to template
        return render_template(
            "index.html",
            prediction_text=result,
            risk_percent=risk_percent
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
