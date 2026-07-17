FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copiar requirements (solo los de la API de demo; no instala torch/mlflow)
COPY dockerfiles/fastapi/requirements.txt /app/requirements.txt

# Instalar dependencias Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar aplicación FastAPI
COPY dockerfiles/fastapi/app.py /app/app.py

# Copiar código fuente del proyecto (para cargar modelos)
COPY src /app/src

# Copiar los scripts: la API reutiliza scripts/evaluation/leaderboard.py
# (build_leaderboard + add_deltas) como única fuente de verdad del ranking.
COPY scripts /app/scripts

EXPOSE 8800
