# !pip install pandas -q
# !pip install tensorflow==2.14.0 -q

import streamlit as st
import joblib

from clases_y_funciones import *

# Se carga el pipeline del modelo.
preprocesamiento = joblib.load('preprocesamiento.joblib')
regresion = joblib.load('regresion.joblib')
clasificacion = joblib.load('clasificacion.joblib')

st.title("Prediccion de lluvia")

features = ["Unnamed: 0", "Date", "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
"WindGustDir", "WindGustSpeed", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
"Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday", "RainTomorrow", "RainfallTomorrow"]

input_features = [ "Date",  "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
"WindGustDir", "WindGustSpeed", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
"Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",  "RainTomorrow", "RainfallTomorrow"]

default_values = {"Unnamed: 0": 0, 
                  "Date": "2008-12-1", 
                  "Location": "Sydney", 
                  "MinTemp": 20, 
                  "MaxTemp": 25, 
                  "Rainfall": 10, 
                  "Evaporation": 10, 
                  "Sunshine": 7, 
                  "WindGustDir": 0, 
                  "WindGustSpeed": 10, 
                  "Humidity9am": 50, 
                  "Humidity3pm": 50, 
                  "Pressure9am": 1010, 
                  "Pressure3pm": 1010, 
                  "Cloud9am": 4, 
                  "Cloud3pm": 4, 
                  "Temp9am": 20, 
                  "Temp3pm": 25, 
                  "RainToday": "No", 
                  "RainTomorrow": 0, 
                  "RainfallTomorrow": 10}



# Definir sliders
date = st.slider("Date", min_value=0, max_value=145411, value=72705)
min_temp = st.slider("MinTemp", min_value=-8.5, max_value=33.9, value=20)
max_temp = st.slider("MaxTemp", min_value=-4.8, max_value=48.1, value=25)
rainfall = st.slider("Rainfall", min_value=0, max_value=371.0, value=10)
evaporation = st.slider("Evaporation", min_value=0, max_value=145.0, value=10)
sunshine = st.slider("Sunshine", min_value=0, max_value=14.5, value=7)
humidity_9am = st.slider("Humidity9am", min_value=0, max_value=100, value=50)
humidity_3pm = st.slider("Humidity3pm", min_value=0, max_value=100, value=50)
pressure_9am = st.slider("Pressure9am", min_value=980.5, max_value=1041.0, value=1010)
pressure_3pm = st.slider("Pressure3pm", min_value=977.1, max_value=1039.6, value=1010)
cloud_9am = st.slider("Cloud9am", min_value=0, max_value=9, value=4)
cloud_3pm = st.slider("Cloud3pm", min_value=0, max_value=9, value=4)
temp_9am = st.slider("Temp9am", min_value=-7.2, max_value=40.2, value=20)
temp_3pm = st.slider("Temp3pm", min_value=-5.4, max_value=46.7, value=25)

#wind_GustDir = st.slider("WindGustDir", min_value=-10, max_value=10, value=0)
#wind_GustSpeed = st.slider("WindGustSpeed", min_value=0, max_value=200, value=10)
#rain_tomorrow = st.slider("RainTomorrow", min_value=0, max_value=1, value=0, step=1)
#rainfall_tomorrow = st.slider("RainfallTomorrow", min_value=0, max_value=371.000000, value=10)

def get_user_input():
    """
    esta función genera los inputs del frontend de streamlit para que el usuario pueda cargar los valores.
    Además, contiene el botón para hacer el submit y obtener la predicción.
    No hace falta hacerlo así, las posibilidades son infinitas.
    """
    input_dict = {}

    with st.form(key='my_form'):
        for feat in input_features:
            input_value = st.number_input(f"Enter value for {feat}", value=0.0, step=0.1)
            input_dict[feat] = input_value

        submit_button = st.form_submit_button(label='Submit')

    return input_dict, submit_button


user_input, submit_button = get_user_input()

if submit_button:
    values = default_values.update(user_input)

    values = pd.DataFrame([values])

    print(values)
    # Preprocesar los datos y hacer la predicción
    #datos = preprocesamiento.transform(user_input)
    #prediction = clasificacion.predict(datos)
    #prediction_value = prediction[0]

    # Display the prediction
    #st.header('Predicción de lluvia:')
    #st.write('Va a llover mañana' if prediction_value > 0.5  else 'No va a llover mañana')