import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo y el escalador
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Título y descripción
st.title("Asistente IA para Cardiólogos")
st.write("""
    Esta aplicación utiliza un modelo de inteligencia artificial basado en K-Nearest Neighbors (KNN) 
    para predecir si una persona tiene o no problemas cardíacos en base a su edad y colesterol. 
    El modelo fue entrenado con un conjunto de datos que contiene estas variables, y los resultados son 
    clasificados en dos categorías: 0 para "No tiene problema cardíaco" y 1 para "Tiene problema cardíaco".
""")

# Información sobre el autor
st.write("Desarrollado por Juan Nicolás Gómez")

# Crear las pestañas usando st.radio
opciones = ["Ingresar Datos", "Realizar Predicción"]
seleccion = st.radio("Selecciona una opción", opciones)

# Pestaña para ingresar datos
if seleccion == "Ingresar Datos":
    st.subheader("Ingresa la información del paciente para realizar la predicción:")

    # Formulario de ingreso de datos
    edad = st.number_input("Edad", min_value=18, max_value=80, value=30)
    colesterol = st.number_input("Colesterol", min_value=50, max_value=600, value=200)

    # Almacenar los datos en el session_state para usar en la predicción
    if st.button("Guardar Datos"):
        st.session_state.edad = edad
        st.session_state.colesterol = colesterol
        st.success("Datos guardados. Ahora puedes ir a la pestaña de predicción.")

# Pestaña para mostrar la predicción
elif seleccion == "Realizar Predicción":
    if "edad" in st.session_state and "colesterol" in st.session_state:
        # Extraer los datos guardados
        edad = st.session_state.edad
        colesterol = st.session_state.colesterol
        
        st.subheader("Resultados de la predicción:")

        # Normalizar los datos
        datos = np.array([[edad, colesterol]])
        datos_normalizados = escalador.transform(datos)

        # Realizar la predicción
        prediccion = modelo_knn.predict(datos_normalizados)

        # Mostrar el resultado
        if prediccion == 0:
            st.success("La predicción es: **No tiene problema cardíaco**")
        else:
            st.error("La predicción es: **Tiene problema cardíaco**")
            st.image("https://www.clikisalud.net/wp-content/uploads/2018/09/problemas-cardiacos-jovenes.jpg", caption="Problemas cardiacos en jóvenes")
    else:
        st.warning("Primero ingresa los datos en la pestaña 'Ingresar Datos'.")
