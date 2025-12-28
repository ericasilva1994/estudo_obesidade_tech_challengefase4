FROM python:3.11-slim

WORKDIR /app

# dependências de sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copiar requirements
COPY requirements.txt .

# instalar dependências
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# copiar aplicação
COPY app ./app

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]
