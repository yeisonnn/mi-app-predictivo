import pandas as pd
import streamlit as st
import pickle

# 1. Título de la aplicación
st.title('Sistema Predictivo de Alertas Tempranas de Reincidencia de Abuso')
st.write('Seleccione las características para evaluar la probabilidad mediante la Red Neuronal.')

# 2. Cargar el modelo guardado
@st.cache_resource # Esto hace que el modelo cargue rápido y no se reinicie a cada clic
def cargar_modelo():
    filename = 'modelo-class.pkl'
    return pickle.load(open(filename, 'rb'))

modelo, labelencoder, variables = cargar_modelo()

# 3. Cargar el dataset original para extraer las categorías exactas
@st.cache_data
def cargar_datos():
    # Asegúrate de que este sea el nombre exacto de tu archivo en GitHub
    return pd.read_csv('dataset_final.xlsx') 

df_base = cargar_datos()

# ⚠️ IMPORTANTE: Escribe aquí el nombre exacto de la columna que tu modelo predice (la 'Y')
# para que no aparezca como una opción en el formulario.
columna_objetivo = 'Reincidencia' 

# 4. Crear el formulario dinámicamente
st.subheader('Ingreso de Datos')
datos_usuario = {}

# Recorremos cada columna del dataset (excepto la que predecimos)
columnas_entrada = [col for col in df_base.columns if col != columna_objetivo]

for col in columnas_entrada:
    # Extraemos las opciones únicas de esa columna directamente del Excel
    opciones_unicas = df_base[col].dropna().unique().tolist()
    # Creamos un menú desplegable para cada variable
    datos_usuario[col] = st.selectbox(f'Seleccione {col}', opciones_unicas)

# 5. Ejecutar la Predicción
if st.button('Generar Predicción'):
    
    # Convertir las selecciones del usuario en un DataFrame
    data_usuario_df = pd.DataFrame([datos_usuario])
    
    # Aplicar One-Hot Encoding (igual que en el entrenamiento)
    data_preparada = pd.get_dummies(data_usuario_df, columns=columnas_entrada, drop_first=False, dtype=int)
    
    # Alinear las columnas: vital para que la red neuronal reciba el formato exacto que espera
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
    
    # Hacer la predicción
    Y_pred = modelo.predict(data_preparada)
    
    # Decodificar el resultado al texto original
    prediccion_texto = labelencoder.inverse_transform(Y_pred)
    
    # Mostrar el resultado
    st.divider()
    st.subheader('🚨 Resultado de la Evaluación:')
    st.success(f"Clasificación estimada por el modelo: **{prediccion_texto[0]}**")