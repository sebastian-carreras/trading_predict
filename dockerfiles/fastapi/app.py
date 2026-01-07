"""
FastAPI Service - Trading Predict

Endpoints:
- GET  /health - Health check
- POST /predict/e1/{ticker} - Predicción E1 (retorno 90 días)
- POST /predict/e3/{ticker} - Predicción E3 (retorno 30 min)
- POST /models/register - Registrar nuevos modelos (desde Airflow)
- GET  /models/status - Estado de modelos cargados
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Optional
import torch
import pandas as pd
import mlflow
import os

app = FastAPI(
    title="Trading Predict API",
    description="API para predicciones de trading con ML",
    version="1.0.0",
)

# Configuración MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Cache de modelos cargados en memoria
models_cache: Dict[str, Dict] = {
    "e1": {},  # {"AAPL": {"model": ..., "scaler": ...}}
    "e3": {},
}

# Estado de modelos registrados
models_registry: Dict[str, Dict] = {
    "e1_conservative": {"last_update": None, "run_dir": None, "tickers": []},
    "e3_intraday": {"last_update": None, "run_dir": None, "tickers": []},
}


class PredictionRequest(BaseModel):
    """Request para predicción."""
    features: Optional[list] = None  # Si se pasan features directamente
    use_latest_data: bool = True     # Usar últimos datos disponibles


class ModelRegistration(BaseModel):
    """Registro de nuevos modelos desde Airflow."""
    strategy: str  # "e1_conservative" | "e3_intraday"
    run_dir: str
    timestamp: str


@app.get("/")
def read_root():
    """Health check básico."""
    return {
        "service": "Trading Predict API",
        "status": "running",
        "strategies": ["e1_conservative", "e3_intraday"],
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


@app.get("/health")
def health_check():
    """Health check detallado."""
    return {
        "status": "healthy",
        "models_loaded": {
            "e1": len(models_cache["e1"]),
            "e3": len(models_cache["e3"]),
        },
        "registry": models_registry,
    }


@app.post("/models/register")
def register_models(registration: ModelRegistration):
    """Registrar nuevos modelos entrenados (llamado por Airflow)."""
    strategy = registration.strategy
    run_dir = Path(registration.run_dir)
    
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run directory not found: {run_dir}")
    
    # Actualizar registry
    if strategy not in models_registry:
        models_registry[strategy] = {}
    
    models_registry[strategy].update({
        "last_update": registration.timestamp,
        "run_dir": str(run_dir),
        "tickers": [p.stem for p in run_dir.glob("*/") if p.is_dir()],
    })
    
    # Opcional: pre-cargar modelos en cache
    # (para producción, cargar bajo demanda)
    
    return {
        "status": "registered",
        "strategy": strategy,
        "tickers": models_registry[strategy]["tickers"],
        "timestamp": registration.timestamp,
    }


@app.get("/models/status")
def get_models_status():
    """Estado de modelos registrados y cargados."""
    return {
        "registry": models_registry,
        "cache": {
            "e1_loaded": list(models_cache["e1"].keys()),
            "e3_loaded": list(models_cache["e3"].keys()),
        },
    }


@app.post("/predict/e1/{ticker}")
def predict_e1(ticker: str, request: PredictionRequest):
    """
    Predicción E1: retorno acumulado a 90 días.
    
    Args:
        ticker: Símbolo del activo (ej: AAPL, YPF)
        request: Configuración de predicción
    
    Returns:
        Predicción de retorno a 90 días
    """
    # Validar ticker registrado
    if ticker not in models_registry.get("e1_conservative", {}).get("tickers", []):
        raise HTTPException(
            status_code=404,
            detail=f"Ticker {ticker} no disponible en E1. Disponibles: {models_registry['e1_conservative'].get('tickers', [])}",
        )
    
    # Cargar modelo si no está en cache
    if ticker not in models_cache["e1"]:
        run_dir = Path(models_registry["e1_conservative"]["run_dir"])
        model_path = run_dir / ticker / f"{ticker}_model.pth"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {model_path}")
        
        # TODO: Cargar modelo GRU desde PyTorch
        # models_cache["e1"][ticker] = {"model": model, "scaler": scaler}
        
        return {
            "ticker": ticker,
            "strategy": "e1_conservative",
            "status": "model_loading_pending",
            "message": "Implementar carga de modelo PyTorch",
        }
    
    # TODO: Hacer predicción con features más recientes
    return {
        "ticker": ticker,
        "strategy": "e1_conservative",
        "predicted_return_90d": 0.0,  # Placeholder
        "confidence": "pending_implementation",
    }


@app.post("/predict/e3/{ticker}")
def predict_e3(ticker: str, request: PredictionRequest):
    """
    Predicción E3: retorno intradía a 30 min.
    
    Args:
        ticker: Símbolo del activo
        request: Configuración
    
    Returns:
        Predicción de retorno a 6 barras (30 min)
    """
    if ticker not in models_registry.get("e3_intraday", {}).get("tickers", []):
        raise HTTPException(
            status_code=404,
            detail=f"Ticker {ticker} no disponible en E3",
        )
    
    # Similar a E1, cargar ensemble LSTM
    return {
        "ticker": ticker,
        "strategy": "e3_intraday",
        "predicted_return_30min": 0.0,  # Placeholder
        "status": "pending_implementation",
    }


@app.get("/mlflow/experiments")
def list_mlflow_experiments():
    """Listar experimentos de MLflow."""
    try:
        experiments = mlflow.search_experiments()
        return {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error consultando MLflow: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)
