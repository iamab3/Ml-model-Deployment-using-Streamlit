# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open('/Users/abhishekjain/Documents/GitHub/ML Projects/Ml-model-Deployment-using-Streamlit/trained_diabetes_model.sav', 'rb'))

# Making a predictive system to predict if a female has diabetes or not
input = (4,110,92,0,0,37.6,0.191,30)

# Chaning the input data to numpy array to make its processing easy
input_np = np.asarray(input)

# Reshaping the array to predict for one instance - to tell the model to just predict for one data point
input_np_reshape = input_np.reshape(1,-1)
input_np_reshape.shape

# Predicting 
prediction = loaded_model.predict(input_np_reshape)
print(prediction)

# Prediction is a list so we need the first element from the list

if (prediction[0]==0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')