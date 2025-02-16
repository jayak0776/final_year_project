from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load the Random Forest CLassifier model
filename = 'heart_disease_model.pkl'
model = pickle.load(open(filename, 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/causes')
def causes():
    return render_template('causes.html')

@app.route('/team')
def team():
    return render_template('team.html')



@app.route('/prediction')
def prediction():
    return render_template('main.html')


# @app.route('/predict', methods=['GET','POST'])
# def predict():
#     if request.method == 'POST':

#         age = int(request.form['age'])
#         sex = request.form.get('sex')
#         cp = request.form.get('cp')
#         trestbps = int(request.form['trestbps'])
#         chol = int(request.form['chol'])
#         fbs = request.form.get('fbs')
#         restecg = int(request.form['restecg'])
#         thalach = int(request.form['thalach'])
#         exang = request.form.get('exang')
#         oldpeak = float(request.form['oldpeak'])
#         slope = request.form.get('slope')
#         ca = int(request.form['ca'])
#         thal = request.form.get('thal')
        
#         data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
#         print(data)
#         my_prediction = model.predict(data)
#         print(my_prediction)
        
#         return render_template('result.html', prediction=my_prediction)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input values from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])

        # Convert to NumPy array and reshape
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Scale the input data
        data_scaled = scaler.transform(data)

        # Predict class and probability
        my_prediction = model.predict(data_scaled)[0]  # 0 (low risk) or 1 (high risk)
        probability = model.predict_proba(data_scaled)[0][1]  # Probability of class 1 (high risk)

        # Convert probability to percentage
        risk_percentage = round(probability * 100, 2)

        return render_template('result.html', prediction=my_prediction, risk_percentage=risk_percentage)
        
        
        

if __name__ == '__main__':
     app.run(debug=True)