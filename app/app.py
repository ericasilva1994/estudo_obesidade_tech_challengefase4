import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Avaliação de Risco de Obesidade",
    layout="wide",
    page_icon="🩺"
)

st.title("🩺 Avaliação Inteligente de Risco de Obesidade")
st.markdown(
    "Aplicação preditiva desenvolvida com **Machine Learning**, "
    "baseada em dados de hábitos de vida e características físicas."
)

# Carregamento dos artefatos treinados
model = joblib.load("app/model.pkl")
encoder = joblib.load("app/encoder.pkl")
scaler = joblib.load("app/scaler.pkl")

# Carregamento da base original para EDA
df_eda = pd.read_excel("data/Obesity.xlsx")

# Tradução das classes
mapa_obesidade = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Grau I",
    "Overweight_Level_II": "Sobrepeso Grau II",
    "Obesity_Type_I": "Obesidade Grau I",
    "Obesity_Type_II": "Obesidade Grau II",
    "Obesity_Type_III": "Obesidade Grau III"
}

df_eda["Classificação"] = df_eda["Obesity"].map(mapa_obesidade)

ordem_classes = [
    "Abaixo do Peso",
    "Peso Normal",
    "Sobrepeso Grau I",
    "Sobrepeso Grau II",
    "Obesidade Grau I",
    "Obesidade Grau II",
    "Obesidade Grau III"
]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 Perfil do Paciente",
    "🏃 Hábitos de Vida",
    "📊 Resultado",
    "📈 Análise Exploratória",
    "📌 Principais Insights"
])

# ABA 1 — Perfil do Paciente
with tab1:
    st.subheader("👤 Perfil do Paciente")

    col1, col2 = st.columns(2)

    with col1:
        genero = st.selectbox("Gênero", ["Masculino", "Feminino"])
        idade = st.number_input("Idade", 14, 100, 30)
        altura = st.number_input("Altura (m)", 1.40, 2.10, 1.70)

    with col2:
        peso = st.number_input("Peso (kg)", 40.0, 200.0, 70.0)
        historico_familiar = st.selectbox("Histórico familiar de obesidade?", ["Sim", "Não"])

# ABA 2 — Hábitos de Vida
with tab2:
    st.subheader("🏃 Hábitos de Vida")

    col1, col2, col3 = st.columns(3)

    with col1:
        favc = st.selectbox("Consumo frequente de alimentos calóricos?", ["Sim", "Não"])
        fcvc = st.slider("Consumo de vegetais (1 = baixo, 3 = alto)", 1, 3, 2)
        ncp = st.slider("Número de refeições por dia", 1, 4, 3)

    with col2:
        caec = st.selectbox("Consumo entre refeições", ["Não", "Às vezes", "Frequentemente", "Sempre"])
        ch2o = st.slider("Consumo diário de água (1 = baixo, 3 = alto)", 1, 3, 2)
        calc = st.selectbox("Consumo de álcool", ["Não", "Às vezes", "Frequentemente"])

    with col3:
        faf = st.slider("Frequência de atividade física", 0.0, 3.0, 1.0)
        tue = st.slider("Tempo diário em eletrônicos", 0.0, 2.0, 1.0)
        mtrans = st.selectbox("Meio de transporte principal",
                              ["Caminhada", "Transporte Público", "Automóvel", "Motocicleta"])

# ABA 3 — Resultado
with tab3:
    st.subheader("📊 Resultado da Avaliação")

    if st.button("🔍 Avaliar risco de obesidade"):
        try:
            imc = peso / (altura ** 2)

            df_input = pd.DataFrame([{
                "Gender": "Male" if genero == "Masculino" else "Female",
                "Age": idade,
                "Height": altura,
                "Weight": peso,
                "family_history": "yes" if historico_familiar == "Sim" else "no",
                "FAVC": "yes" if favc == "Sim" else "no",
                "FCVC": fcvc,
                "NCP": ncp,
                "CAEC": caec,
                "SMOKE": "no",
                "CH2O": ch2o,
                "SCC": "no",
                "FAF": faf,
                "TUE": tue,
                "CALC": calc,
                "MTRANS": mtrans,
                "BMI": imc
            }])

            cat_cols = encoder.feature_names_in_
            num_cols = scaler.feature_names_in_

            X_cat = encoder.transform(df_input[cat_cols])
            X_num = scaler.transform(df_input[num_cols])

            X_final = np.hstack([X_num, X_cat])
            pred = model.predict(X_final)[0]

            st.success(f"🧠 Classificação prevista: **{mapa_obesidade[pred]}**")

        except Exception as e:
            st.error("Erro ao processar os dados.")
            st.exception(e)

# ABA 4 — Análise Exploratória
with tab4:
    st.subheader("📈 Análise Exploratória dos Dados")

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df_eda, x="Classificação", order=ordem_classes, palette="Blues", ax=ax1)
    ax1.set_title("Distribuição dos níveis de obesidade")
    ax1.set_xlabel("Classificação")
    ax1.set_ylabel("Quantidade de pessoas")
    ax1.tick_params(axis="x", rotation=30)
    st.pyplot(fig1)

    df_eda["IMC"] = df_eda["Weight"] / (df_eda["Height"] ** 2)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_eda, x="Classificação", y="IMC",
                order=ordem_classes, palette="Set2", ax=ax2)
    ax2.set_title("Distribuição do IMC por nível de obesidade")
    ax2.set_xlabel("Classificação")
    ax2.set_ylabel("IMC")
    ax2.tick_params(axis="x", rotation=30)
    st.pyplot(fig2)

# ABA 5 — Principais Insights
with tab5:
    st.subheader("📌 Principais Insights do Estudo")

    st.markdown("""
    **1️⃣ Distribuição equilibrada dos dados**  
    O conjunto de dados apresenta boa representatividade entre os níveis de obesidade, 
    reduzindo viés no treinamento do modelo.

    **2️⃣ IMC como principal fator discriminante**  
    Há uma clara progressão do IMC conforme o avanço dos níveis de obesidade, 
    validando sua relevância clínica.

    **3️⃣ Atividade física influencia diretamente o risco**  
    Indivíduos com menor frequência de atividade física tendem a apresentar 
    níveis mais elevados de obesidade.

    **4️⃣ Hábitos alimentares impactam fortemente a classificação**  
    Consumo frequente de alimentos calóricos e alimentação entre refeições 
    aparecem associados a maiores riscos.

    **5️⃣ Modelo com alto desempenho preditivo**  
    O modelo Random Forest alcançou aproximadamente **92% de acurácia**, 
    demonstrando excelente capacidade de generalização.
    """)
