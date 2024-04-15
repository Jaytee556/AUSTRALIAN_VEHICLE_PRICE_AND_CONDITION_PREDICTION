import streamlit as st
import joblib
import numpy as np

# Load the trained model
classifier = joblib.load('classifier.pkl')

st.cache_resource
def predict_car_condition(Year, CarOrSUV, Kilometres, Price, ColorExtInt):
    # Pre-processing User input
    #input for colour 
    if ColorExtInt == "Black/Black":
        ColorExtInt = 0
    elif ColorExtInt == "Grey/Black":
        ColorExtInt = 1
    elif ColorExtInt == "White/Black":
        ColorExtInt = 2
    elif ColorExtInt == "White/Brown":
        ColorExtInt = 3
    else:
        ColorExtInt == 4


    #input for Car/SUV
    if CarOrSUV == "SUV":
        CarOrSUV= 0
    elif CarOrSUV == "Coupe":
        CarOrSUV = 1
    elif CarOrSUV == "Hatchback":
        CarOrSUV = 2   
    elif CarOrSUV == "Sedan":
        CarOrSUV = 3
    elif CarOrSUV == "Alto Blacktown MG":
        CarOrSUV = 4
    else:
        CarOrSUV = 5



    # Making Predictions
    prediction = classifier.predict([[Year, CarOrSUV, Kilometres, Price, ColorExtInt]])

    if prediction == 0:
        pred = "NEW CAR"
    else:
        pred = "USED CAR"

    return pred

def main():

    # Add a title in blue color
    st.markdown("<h1 style='color: blue;font-size: 34px;'>Australian Vehicle Condition Prediction App</h1>", unsafe_allow_html=True)
    
    
    
    Year = st.number_input('Year of Car Manufacture', value = 0, step = 1)
    CarOrSUV = st.selectbox("Type of Car", ("SUV", "Coupe", "Hatchback", "Sedan", "Alto Blacktown MG"))
    Kilometres = st.number_input('Kilometres')
    ColorExtInt = st.selectbox("Color of Car", ("Black/Black", "Grey/Black", "White/Black", "White/Brown"))
    Price = st.number_input('Price (in Australian Dollars($)')


    if st.button('Predict'):
        prediction = predict_car_condition(Year, CarOrSUV, Kilometres, Price, ColorExtInt)
        st.write('Condition of the Vehicle:', prediction)
        print(prediction)

if __name__ == '__main__':
    main()
