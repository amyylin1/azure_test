# import libraris
import pandas as pd

import numpy as np
from flask import Flask, render_template, request
import pickle

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

XX = [2.54, 85, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 1]


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
    prediction = model.predict_proba( df_XX )
    print(prediction)
    #prediction = model.predict(final_features)
    output = np.round(prediction[0][1], 2)
    print( 'You are likely: {}'.format(output) )
    return render_template('index.html', prediction_text='Probability: {}'.format(output))

# start the flask server
if __name__ == '__main__':
    app.run(debug=True)