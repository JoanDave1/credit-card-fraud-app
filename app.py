from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = np.array([[
        float(request.form['distance_from_home']),
        float(request.form['distance_from_last_transaction']),
        float(request.form['ratio_to_median_purchase_price']),
        int(request.form['online_order']),
        int(request.form['used_chip']),
        int(request.form['used_pin_number']),
        int(request.form['repeat_retailer'])
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    result = "Fraudulent Transaction 🚨" if prediction == 1 else "Legitimate Transaction ✅"

    return render_template(
        'index.html',
        prediction=result,
        probability=round(probability * 100, 2)
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)


