from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('predict.html')  # Redirect to the predict.html template

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    region = int(request.form['region'])

    # Make a prediction
    prediction = model.predict([[age, sex, bmi, children, smoker, region]])

    return render_template('predict.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)    
    #app.run(debug=True)

