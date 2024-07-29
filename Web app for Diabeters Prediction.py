#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:02:54 2024

@author: abhishekjain
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('/Users/abhishekjain/Documents/GitHub/ML Projects/Ml-model-Deployment-using-Streamlit/trained_diabetes_model.sav', 'rb'))

# Creating a function for prediction
def diabetes_prediction(input_data):

    # Chaning the input data to numpy array to make its processing easy
    input_np = np.asarray(input)

    # Reshaping the array to predict for one instance - to tell the model to just predict for one data point
    input_np_reshape = input_np.reshape(1,-1)

    # Predicting 
    prediction = loaded_model.predict(input_np_reshape)
    print(prediction)

    # Prediction is a list so we need the first element from the list

    if (prediction[0]==0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    
    
    
def main():
    
    # Giving a title
    st.title('Web App for Diabetic Prediction')
    
    # Creating inout data fields and getting data from user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Blood Glucose levels')
    BloodPressure = st.text_input('Blood Pressure values')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')
    
    # Code for prediction
    Diagnostic = ''
    
    # Creating a button for prediction
    if st.button('Diabetes Run Test'):
        Diagnostic = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(Diagnostic)
    

if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    