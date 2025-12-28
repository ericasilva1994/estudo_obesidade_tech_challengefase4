import streamlit as st
import pandas as pd
import joblib

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Avaliação de Risco de Obesidade",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Avaliação de Risco de Obesidade")
st.markdown(
    "Aplicação preditiva desenvolvida com **Machine Learning** para avaliação do risco de obesidade."
)

st.divider()

# =========================================================
# CARREGAMENTO DOS ARTEFATOS
# =========================================================
@st.cache_resource
def load_models():
    model = joblib.load("app/model.pkl")
    encoder = joblib.load("app/encoder.pkl")
    scaler = joblib.load("app/scaler.pkl")
    return model, encoder, scaler

model, encoder, scaler = load_models()

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "👤 Perfil do Paciente",
    "🏃 Hábitos de Vida",
    "📊 Resultado"
])

# =========================================================
# TAB 1 — PERFIL DO PACIENTE
# =========================================================
with tab1:
    st.subheader("👤 Perfil do Paciente")

    col1, col2 = st.columns(2)

    with col1:
        genero = st.selectbox("Gênero", ["Masculino", "Feminino"])
        idade = st.number_input("Idade", min_value=14, max_value=100, value=30)
        altura = st.number_input("Altura (m)", min_value=1.40, max_value=2.20, value=1.70)

    with col2:
        peso = st.number_input("Peso (kg)", min_value=40.0, max_value=200.0, value=70.0)
        historico_familiar = st.selectbox(
            "Histórico familiar de obesidade?",
            ["Sim", "Não"]
        )

# =========================================================
# TAB 2 — HÁBITOS DE VIDA
# =========================================================
with tab2:
    st.subheader("🏃 Hábitos de Vida")

    col3, col4 = st.columns(2)

    with col3:
        favc = st.selectbox(
            "Consome alimentos altamente calóricos?",
            ["Sim", "Não"]
        )
        fcvc = st.slider(
            "Consumo de vegetais",
            1, 3, 2,
            help="1 = baixo | 3 = alto"
        )
        ncp = st.slider(
            "Número de refeições por dia",
            1, 4, 3
        )
        caec = st.selectbox(
            "Come entre as refeições?",
            ["Não", "Às vezes", "Frequentemente", "Sempre"]
        )
        fuma = st.selectbox("Fuma?", ["Sim", "Não"])

    with col4:
        agua = st.slider(
            "Consumo diário de água (litros)",
            1.0, 3.0, 2.0
        )
        monitora_calorias = st.selectbox(
            "Monitora consumo de calorias?",
            ["Sim", "Não"]
        )
        atividade_fisica = st.slider(
            "Atividade física (dias por semana)",
            0, 7, 2
        )
        tempo_tela = st.slider(
            "Uso de tecnologia (horas por dia)",
            0.0, 6.0, 1.0
        )
        alcool = st.selectbox(
            "Consumo de álcool",
            ["Não", "Às vezes", "Frequentemente"]
        )

        transporte = st.selectbox(
            "Meio de transporte principal",
            [
                "Automóvel",
                "Moto",
                "Bicicleta",
                "Transporte Público",
                "Caminhada"
            ]
        )

# =========================================================
# TAB 3 — RESULTADO
# =========================================================
with tab3:
    st.subheader("📊 Resultado da Avaliação")

    if st.button("🔍 Avaliar risco de obesidade", use_container_width=True):
        try:
            # =================================================
            # DATAFRAME COM COLUNAS IDÊNTICAS AO TREINO
            # =================================================
            df = pd.DataFrame([{
                "Gender": "Male" if genero == "Masculino" else "Female",
                "Age": idade,
                "Height": altura,
                "Weight": peso,
                "family_history": "yes" if historico_familiar == "Sim" else "no",
                "FAVC": "yes" if favc == "Sim" else "no",
                "FCVC": fcvc,
                "NCP": ncp,
                "CAEC": caec,
                "SMOKE": "yes" if fuma == "Sim" else "no",
                "CH2O": agua,
                "SCC": "yes" if monitora_calorias == "Sim" else "no",
                "FAF": atividade_fisica,
                "TUE": tempo_tela,
                "CALC": alcool,
                "MTRANS": transporte
            }])

            # =================================================
            # TRANSFORMAÇÕES
            # =================================================
            df_encoded = encoder.transform(df)
            df_scaled = scaler.transform(df_encoded)

            # =================================================
            # PREDIÇÃO
            # =================================================
            resultado = model.predict(df_scaled)[0]

            st.success("✅ Avaliação concluída com sucesso!")
            st.metric(
                label="Classificação de Risco de Obesidade",
                value=resultado
            )

        except Exception as e:
            st.error("❌ Erro ao processar os dados.")
            st.exception(e)

st.divider()
st.caption("Projeto acadêmico • Streamlit + Machine Learning • Python 3.11")





