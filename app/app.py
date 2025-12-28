# Aplicação preditiva de obesidade usando Streamlit


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Carrega o modelo treinado
model = joblib.load("model.pkl")

# Carrega o encoder das variáveis categóricas
encoder = joblib.load("encoder.pkl")

# Carrega o scaler das variáveis numéricas
scaler = joblib.load("scaler.pkl")

# Configuração da página
st.set_page_config(
    page_title="Predição de Obesidade",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Sistema Preditivo de Obesidade")
st.write(
    "Este sistema utiliza Machine Learning para auxiliar a equipe médica "
    "na identificação do nível de obesidade de um paciente."
)
# Formulário de entrada de dados
st.header("📋 Dados do paciente")

gender = st.selectbox("Gênero", ["Male", "Female"])
age = st.number_input("Idade", min_value=14, max_value=100, value=30)
height = st.number_input("Altura (em metros)", min_value=1.40, max_value=2.10, value=1.70)
weight = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)

family_history = st.selectbox("Histórico familiar de excesso de peso?", ["yes", "no"])
favc = st.selectbox("Consome alimentos altamente calóricos?", ["yes", "no"])
fcvc = st.slider("Consumo de vegetais", 1, 3, 2)
ncp = st.slider("Número de refeições principais", 1, 4, 3)
caec = st.selectbox("Come entre as refeições?", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Fuma?", ["yes", "no"])
ch2o = st.slider("Consumo diário de água", 1, 3, 2)
scc = st.selectbox("Monitora calorias?", ["yes", "no"])
faf = st.slider("Frequência de atividade física", 0, 3, 1)
tue = st.slider("Tempo usando eletrônicos", 0, 2, 1)
calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox(
    "Meio de transporte",
    ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
)
# Botão de predição

if st.button("🔍 Realizar predição"):

    # Criação do DataFrame com os dados do usuário
    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans,
        "BMI": weight / (height ** 2)
    }])

    # Separação de colunas categóricas e numéricas
    cat_cols = input_data.select_dtypes(include="object").columns
    num_cols = input_data.select_dtypes(exclude="object").columns

    # Aplicação do encoder nas variáveis categóricas
    X_cat = encoder.transform(input_data[cat_cols])

    # Aplicação do scaler nas variáveis numéricas
    X_num = scaler.transform(input_data[num_cols])

    # Junção das variáveis
    X_final = np.hstack([X_num, X_cat])

    # Predição do modelo
    prediction = model.predict(X_final)[0]

    # Resultado

    st.success(f"✅ Nível de obesidade previsto: **{prediction}**")

      


