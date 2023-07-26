# import libraris
import pandas as pd

import numpy as np
from flask import Flask, render_template, request
import pickle

## col name
col_names = ['Poverty_Ratio',
            'Age',
            'Categorical_BMI_Healthy_Weight',
            'Categorical_BMI_Obese',
            'Categorical_BMI_Overweight',
            'Categorical_BMI_Underweight',
            'Categorical_BMI_Unknown',
            'Education_12th_Grade_no_diploma',
            'Education_Associates_Academic_Program',
            'Education_Associates_Occupational_Technical_Vocational',
            'Education_Bachelor',
            'Education_Dont_Know',
            'Education_GED_Equivalent',
            'Education_Grade_1-11',
            'Education_Greater_Than_Master',
            'Education_High_School_Graduate',
            'Education_Masters',
            'Education_Refused',
            'Education_Some_College_no_degree',
            'Race_AIAN_AND_other',
            'Race_AIAN_Only',
            'Race_African_American_Only',
            'Race_Asian_Only',
            'Race_Dont_Know',
            'Race_Not_Ascertained',
            'Race_Other',
            'Race_Refused',
            'Race_White_Only']


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

    # for rending result on html GUI
    int_features = int( request.form['options'] )
    print("feature",int_features)

    int_features1 = int( request.form['options1'] )
    print("feature1",int_features1)

    #print("feature",int_features)
    X = np.zeros( len(col_names) )
    X[ int_features ] = 1.0
    print("X",X)
    df_XX = pd.DataFrame(data=[dict(zip(col_names, X) ) ] )

    prediction = model.predict_proba( df_XX )
    print("prediction",prediction)
    
    output = np.round(prediction[0][1], 2)

    print( 'You are likely: {}'.format(output) )
    if output > (.65):
        page = "sad.html"
    else:
        page = "happy.html"
    return render_template(page, prediction_text='Probability: {}'.format(output))

# start the flask server
if __name__ == '__main__':
    app.run(debug=True)