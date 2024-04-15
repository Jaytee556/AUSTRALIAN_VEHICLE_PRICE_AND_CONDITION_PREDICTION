import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('prediction.pkl')

st.cache_resource
def predict_price(Brand, Year, Model, CarOrSUV, UsedOrNew, Transmission, DriveType, FuelType, Kilometres, CylindersinEngine, BodyType, Doors, Seats, Engine_Volumes_Litres, Fuel_Consumption_Litres):
    # Pre-processing User input
    if Transmission == "Automatic":
        Transmission = 0
    else: 
        Transmission = 1

    #input for Condition of Car
    if UsedOrNew == "DEMO":
        UsedOrNew = 0
    elif UsedOrNew == "NEW":
        UsedOrNew = 1
    else:
        UsedOrNew = 2
    
    #Input for Car Brand
    if Brand == "Ford":
        Brand = 0
    elif Brand == "Hyundai":
        Brand = 1
    elif Brand == "Holden":
        Brand = 2
    elif Brand == "Kia":
        Brand = 3
    elif Brand == "Mazda":
        Brand = 4
    elif Brand == "Mercedes-Benz":
        Brand = 5
    elif Brand == "Mitsubishi":
        Brand = 6
    elif Brand == "Nissan":
        Brand = 7
    elif Brand == "Volkswagen":
        Brand = 8
    else:
        Brand = 9
    
    #input for model 
    if Model == "GLC250":
        Model = 0
    elif Model == "Rexton":
        Model = 1
    elif Model == "MG3":
        Model = 2
    elif Model == "Corolla":
        Model = 3
    elif Model == "Hilux":
        Model = 4
    else:
        Model == 5

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

     #input for Drive Type
    if DriveType == "Front":
        DriveType = 0
    elif DriveType == "AWD":
        DriveType = 1
    elif DriveType == "4WD":
        DriveType = 2   
    elif DriveType == "Rear":
        DriveType = 3
    elif DriveType == "Other":
        DriveType = 4
    else:
        DriveType = 5

    #input for Fuel Type
    if FuelType == "Unleaded":
        FuelType = 0   
    elif FuelType == "Diesel":
        FuelType = 1
    elif FuelType == "Premium":
        FuelType = 2   
    elif FuelType == "Hybrid":
        FuelType = 3
    elif FuelType == "Electric":
        FuelType = 4
    elif FuelType == "Other":
        FuelType = 5
    elif FuelType == "LPG":
        FuelType = 6
    elif FuelType == "Leaded":
        FuelType = 7
    else:
        FuelType = 8


    #input for Cylinders in Engine
    if CylindersinEngine == "4 cyl":
        CylindersinEngine = 0
    elif CylindersinEngine == "6 cyl":
        CylindersinEngine = 1
    elif CylindersinEngine == "8 cyl":
        CylindersinEngine = 2
    elif CylindersinEngine == "5 cyl":
        CylindersinEngine = 3
    elif CylindersinEngine == "3 cyl":
        CylindersinEngine = 4
    elif CylindersinEngine == "12 cyl":
        CylindersinEngine = 5
    elif CylindersinEngine == "2 cyl":
        CylindersinEngine = 6
    elif CylindersinEngine == "10 cyl":
        CylindersinEngine = 7
    else:
        CylindersinEngine = 8

    #input for Drive Type
    if BodyType == "SUV":
         BodyType = 0
    elif BodyType == "Hatchback":
        BodyType = 1
    elif Bodyype == "Ute / Tray":
        BodyType = 2   
    elif BodyType == "Sedan":
        BodyType = 3
    elif BodyType == "Wagon":
        BodyType = 4
    elif BodyType == "Commercial":
        BodyType = 5
    elif BodyType == "Coupe":
        BodyType = 6
    elif BodyType == "Convertible":
        BodyType = 7
    elif BodyType == "Other":
        BodyType = 8
    elif BodyType == "People Mover":
        BodyType = 9
    else:
        BodyType = 10
    
    features = np.array([Brand, Year, Model, CarOrSUV, UsedOrNew, Transmission, DriveType, FuelType, Kilometres, CylindersinEngine, BodyType, Doors, Seats, Engine_Volumes_Litres, Fuel_Consumption_Litres]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction

def main():
    # Add a title in blue color
    st.markdown("<h1 style='color: blue;font-size: 36px;'>Australian Vehicle Price Prediction App</h1>", unsafe_allow_html=True)
    
    Brand = st.selectbox("Type of Car Brand", ("Ford", "Hyundai", "Holden", "Kia", "Mazda", "Mercedes-Benz", "Mitsubishi", "Nissan", "Volkswagen"))
    Year = st.number_input('Year of Car Manufacture', value = 0, step = 1)
    Model = st.selectbox("Model of the Vehicle", ("GLC250", "Rexton", "MG3", "Corolla", "Hilux"))
    CarOrSUV = st.selectbox("Type of Car", ("SUV", "Coupe", "Hatchback", "Sedan", "Alto Blacktown MG"))
    UsedOrNew = st.selectbox("Car Condition", ("DEMO", "NEW", "USED"))
    Transmission = st.selectbox('Transmission Type', ("Automatic", "Manual"))
    DriveType = st.selectbox("Drive Type of Vehicle", ("Front", "AWD", "4WD", "Rear", "Other"))
    FuelType = st.selectbox("Fuel Type of the Vehicle", ("Unleaded", "Diesel", "Premium", "Hybrid", "Electric", "Other", "LPG", "Leaded"))
    Kilometres = st.number_input('Kilometres')
    CylindersinEngine = st.selectbox("Number of cylinders in Engine (mote: input e.g 2 cyl, 3 cyl...)", ("4 cyl", "6 cyl", "8 cyl", "5 cyl", "3 cyl", "12 cyl", "2 cyl", "10 cyl"))
    BodyType = st.selectbox("Body Type of Vehicle", ("SUV", "Hatchback", "Ute / Tray", "Sedan", "Wagon", "Commercial", "Coupe", "Convertible", "Other", "People Mover"))
    Doors = st.number_input('Number of Doors (1 - 5)', value = 0, step = 1)
    Seats = st.number_input('Number of Seats (1 - 5)', value = 0, step = 1)
    Engine_Volumes_Litres = st.number_input("Engine_Volumes in litres")
    Fuel_Consumption_Litres = st.number_input("Fuel consumption")
    

    if st.button('Predict'):
        prediction = predict_price(Brand, Year, Model, CarOrSUV, UsedOrNew, Transmission, DriveType, FuelType, Kilometres, CylindersinEngine, BodyType, Doors, Seats, Engine_Volumes_Litres, Fuel_Consumption_Litres)
        st.write('The predicted price of the Vehicle is:', prediction)
        print(prediction)

if __name__ == '__main__':
    main()
