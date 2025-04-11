# web_app/app.py

from flask import Flask, request, render_template
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("../models/best_model.pkl")
le_drug = joblib.load("../models/le_drug.pkl")
le_condition = joblib.load("../models/le_condition.pkl")
# le_effects = joblib.load("../models/le_effects.pkl")  # Not needed anymore

# Load side effect mapping
mapping_df = pd.read_csv("../data/side_effect_mapping.csv")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        drug_name = request.form['drug_name']
        condition = request.form['condition']

        # Encode inputs
        encoded_drug = le_drug.transform([drug_name])[0]
        encoded_condition = le_condition.transform([condition])[0]

        # Predict
        pred = model.predict([[encoded_drug, encoded_condition]])[0]

        # Get readable side effect text
        side_effect = mapping_df[mapping_df['encoded'] == pred]['text'].values[0]

        return render_template('index.html', prediction=side_effect)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
