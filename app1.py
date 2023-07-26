# import libraris
import pandas as pd

import numpy as np
from flask import Flask, render_template, request
import pickle

## dummy data
col_names = [
    'Categorical_BMI_Healthy_Weight',
    'Categorical_BMI_Obese',
    'Categorical_BMI_Overweight',
    'Categorical_BMI_Underweight',
    'Categorical_BMI_Unknown']

XX = [0, 0, 1, 0, 0]


# initialzie the flask app
app = Flask(__name__)
app

# load ml model
model = pickle.load(open('model.pkl', 'rb'))

# define the app route for the default page of the web-app
@app.route('/')
def home():
    return render_template('index.html')

# to use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():

    # for rending result on html gui    
    #int_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    
    df_XX = pd.DataFrame(data=[dict(zip(col_names, XX) ) ] )

    int_features = request.form['options']
    prediction = model.predict_proba( df_XX )
    print(prediction)
    #prediction = model.predict(final_features)
    output = np.round(prediction[0][1], 2)
    print( 'You are likely: {}'.format(output) )
    return render_template('index.html', prediction_text='Probability: {}'.format(output))

# start the flask server
if __name__ == '__main__':
    app.run(debug=True)