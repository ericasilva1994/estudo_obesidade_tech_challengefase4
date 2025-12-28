# ===============================
# Aplicação Streamlit
# Predição de Obesidade
# ===============================

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------
# Caminho base (pasta app)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# -------------------------------
# Carregamento dos artefatos
# -------------------------------
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------------------
# Configuração da página
# -------------------------------
st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Sistema Preditivo de Obesidade")

st.write(
    "Este sistema utiliza Machine Learning para auxiliar a equipe médica "
    "na identificação do nível de obesidade de um paciente."
)

# -------------------------------
# Entrada de dados
# -------------------------------
st.header("📋 Dados do paciente")

gender = st.selectbox("Gênero", ["Male", "Female"])
age = st.number_input("Idade", 14, 100, 30)
height = st.number_input("Altura (m)", 1.40, 2.10, 1.70)
weight = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)

family_history = st.selectbox("Histórico familiar de obesidade?", ["yes", "no"])
favc = st.selectbox("Consome alimentos altamente calóricos?", ["yes", "no"])
fcvc = st.slider("Consumo de vegetais", 1, 3, 2)
ncp = st.slider("Número de refeições", 1, 4, 3)
caec = st.selectbox("Come entre refeições?", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Fuma?", ["yes", "no"])
ch2o = st.slider("Consumo de água", 1, 3, 2)
scc = st.selectbox("Monitora calorias?", ["yes", "no"])
faf = st.slider("Atividade física", 0, 3, 1)
tue = st.slider("Tempo em eletrônicos", 0, 2, 1)
calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox(
    "Meio de transporte",
    ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"]
)

# -------------------------------
# Predição
# -------------------------------
if st.button("🔍 Realizar predição"):

    bmi = weight / (height ** 2)

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
        "BMI": bmi
    }])

    # Separação de colunas
    cat_cols = input_data.select_dtypes(include="object").columns
    num_cols = input_data.select_dtypes(exclude="object").columns

    # Transformações
    X_cat = encoder.transform(input_data[cat_cols])
    X_num = scaler.transform(input_data[num_cols])

    # Junta tudo
    X_final = np.hstack((X_num, X_cat))

    # Predição
    prediction = model.predict(X_final)[0]

    # Resultado
    st.success(f"✅ Nível de obesidade previsto: **{prediction}**")

    st.caption(
        "⚠️ Este resultado é apenas um apoio à decisão e não substitui avaliação médica."
    )


