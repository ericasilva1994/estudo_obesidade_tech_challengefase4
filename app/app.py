# ============================================
# Aplicação Streamlit – Predição de Obesidade
# ============================================

import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------------
# Configuração da página
# --------------------------------------------
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

# --------------------------------------------
# Caminho correto dos arquivos (Streamlit Cloud)
# --------------------------------------------
CURRENT_DIR = os.getcwd()

# (debug visual – pode remover depois)
# st.write("Diretório atual:", CURRENT_DIR)
# st.write("Arquivos:", os.listdir(CURRENT_DIR))

# --------------------------------------------
# Carregamento dos artefatos treinados
# --------------------------------------------
model = joblib.load(os.path.join(CURRENT_DIR, "model.pkl"))
encoder = joblib.load(os.path.join(CURRENT_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(CURRENT_DIR, "scaler.pkl"))

# --------------------------------------------
# Formulário de entrada
# --------------------------------------------
st.header("📋 Dados do paciente")

gender = st.selectbox("Gênero", ["Male", "Female"])
age = st.number_input("Idade", min_value=14, max_value=100, value=30)
height = st.number_input("Altura (m)", min_value=1.40, max_value=2.10, value=1.70)
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

# --------------------------------------------
# Predição
# --------------------------------------------
if st.button("🔍 Realizar predição"):

    # Criação do dataframe de entrada
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

    # Separação de colunas
    cat_cols = input_data.select_dtypes(include="object").columns
    num_cols = input_data.select_dtypes(exclude="object").columns

    # Transformações
    X_cat = encoder.transform(input_data[cat_cols])
    X_num = scaler.transform(input_data[num_cols])

    # Junção final
    X_final = np.hstack([X_num, X_cat])

    # Predição
    prediction = model.predict(X_final)[0]

    # Resultado
    st.success(f"✅ Nível de obesidade previsto: **{prediction}**")

    st.caption(
        "⚠️ Este resultado é apenas um apoio à decisão e não substitui "
        "avaliação médica profissional."
    )
