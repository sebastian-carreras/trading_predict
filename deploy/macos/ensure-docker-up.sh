#!/bin/bash
# Asegura que Docker Desktop esté corriendo y levanta el stack del proyecto.
# Pensado para dispararse cada mañana (LaunchAgent) tras el wake de pmset,
# antes de que los DAGs de E1/E2 arranquen a las 12:00 UTC (09:00 ART).
set -u

# launchd corre con un PATH mínimo: incluir dónde vive el binario docker.
export PATH="/opt/homebrew/bin:/usr/local/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"

# Ruta del proyecto. Debe pasarse por la variable de entorno TRADING_PREDICT_DIR
# (en el .plist, vía EnvironmentVariables — ver README). Se falla si no está.
PROJECT_DIR="${TRADING_PREDICT_DIR:?Definí TRADING_PREDICT_DIR con la ruta del repo trading_predict}"
LOG="/tmp/ensure-docker-up.log"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$LOG"; }

log "==== ensure-docker-up ===="

# 1) ¿Responde el daemon? Si no, abrir Docker Desktop (GUI, en background).
if ! docker info >/dev/null 2>&1; then
  log "Docker daemon no responde; abriendo Docker Desktop..."
  open --background -a Docker || open --background -a "Docker Desktop"
fi

# 2) Esperar hasta ~120s a que el daemon quede listo.
ready=0
for i in $(seq 1 40); do
  if docker info >/dev/null 2>&1; then
    ready=1
    log "Docker daemon listo tras ~$((i * 3))s"
    break
  fi
  sleep 3
done

if [ "$ready" -ne 1 ]; then
  log "ERROR: el daemon de Docker no quedó listo tras el timeout"
  exit 1
fi

# 3) Levantar el stack (idempotente: si ya está arriba, no hace nada).
cd "$PROJECT_DIR" || { log "ERROR: no existe $PROJECT_DIR"; exit 1; }
log "docker compose --profile all up -d ..."
docker compose --profile all up -d >> "$LOG" 2>&1
log "Estado de contenedores:"
docker compose --profile all ps >> "$LOG" 2>&1
log "Hecho."
