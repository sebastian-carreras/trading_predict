#!/usr/bin/env python3
"""
Sistema de evaluación continua para modelos E1 Simple.

Funcionalidad:
1. Guarda predicciones diarias en MLflow
2. Evalúa automáticamente cuando se cumple el horizonte (90 días)
3. Calcula métricas rolling (IC, MAE, Sharpe)
4. Compara con métricas originales del entrenamiento
5. Detecta degradación del modelo (signal decay)

Integración con MLflow:
- Experimento: "E1_Simple_Production_Tracking"
- Cada predicción se guarda como un run
- Evaluaciones se guardan en run de "evaluation"
"""

import os
import sys
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
import argparse
import json

# Importar utils del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import load_yaml


def get_mlflow_tracking_uri(use_docker: bool = False) -> str:
    """Obtener URI de MLflow según ambiente."""
    if use_docker:
        return "http://localhost:5050"
    else:
        # Local SQLite
        return "sqlite:///runs/mlflow_local/mlflow.db"


def save_daily_prediction(
    ticker: str,
    prediction: float,
    features: dict,
    model_run_id: str,
    mlflow_uri: str
) -> str:
    """
    Guarda una predicción diaria en MLflow.
    
    Args:
        ticker: Símbolo del activo
        prediction: Retorno predicho (90 días forward)
        features: Dict con features calculadas
        model_run_id: ID del run de MLflow del modelo usado
        mlflow_uri: URI de MLflow
    
    Returns:
        Run ID de la predicción guardada
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("E1_Simple_Production_Tracking")
    
    timestamp = datetime.now()
    horizon_days = 90
    evaluation_date = timestamp + timedelta(days=horizon_days)
    
    with mlflow.start_run(run_name=f"prediction_{ticker}_{timestamp.strftime('%Y%m%d')}") as run:
        # Parámetros
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("prediction_date", timestamp.strftime("%Y-%m-%d"))
        mlflow.log_param("evaluation_date", evaluation_date.strftime("%Y-%m-%d"))
        mlflow.log_param("horizon_days", horizon_days)
        mlflow.log_param("model_run_id", model_run_id)
        mlflow.log_param("status", "pending")  # pending, evaluated
        
        # Predicción
        mlflow.log_metric("predicted_return", prediction)
        
        # Features (primeras 10)
        for i, (k, v) in enumerate(list(features.items())[:10]):
            if isinstance(v, (int, float)) and np.isfinite(v):
                mlflow.log_metric(f"feature_{k}", float(v))
        
        # Guardar features completas como artifact
        features_df = pd.DataFrame([features])
        features_path = Path(f"/tmp/{ticker}_features_{timestamp.strftime('%Y%m%d')}.csv")
        features_df.to_csv(features_path, index=False)
        mlflow.log_artifact(str(features_path), artifact_path="features")
        features_path.unlink()  # Limpiar
        
        print(f"✓ Predicción guardada: {ticker} = {prediction:+.2%} (eval: {evaluation_date.strftime('%Y-%m-%d')})")
        print(f"  MLflow run ID: {run.info.run_id}")
        
        return run.info.run_id


def get_pending_predictions(ticker: str, mlflow_uri: str) -> pd.DataFrame:
    """
    Obtiene predicciones pendientes de evaluación (>90 días atrás).
    
    Returns:
        DataFrame con predicciones pendientes
    """
    mlflow.set_tracking_uri(mlflow_uri)
    
    experiment = mlflow.get_experiment_by_name("E1_Simple_Production_Tracking")
    if not experiment:
        return pd.DataFrame()
    
    # Buscar runs del ticker con status=pending
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.ticker = '{ticker}' AND params.status = 'pending'",
        order_by=["params.prediction_date ASC"]
    )
    
    if runs.empty:
        return pd.DataFrame()
    
    today = datetime.now()
    evaluable = []
    
    for _, run in runs.iterrows():
        pred_date = datetime.strptime(run["params.prediction_date"], "%Y-%m-%d")
        eval_date = datetime.strptime(run["params.evaluation_date"], "%Y-%m-%d")
        
        # Solo evaluar si ya pasó la fecha de evaluación
        if today >= eval_date:
            evaluable.append({
                'run_id': run['run_id'],
                'prediction_date': pred_date,
                'evaluation_date': eval_date,
                'predicted_return': run['metrics.predicted_return'],
                'model_run_id': run['params.model_run_id']
            })
    
    return pd.DataFrame(evaluable)


def evaluate_predictions(
    ticker: str,
    predictions_df: pd.DataFrame,
    mlflow_uri: str
) -> dict:
    """
    Evalúa predicciones contra retornos reales.
    
    Args:
        ticker: Símbolo
        predictions_df: DataFrame con predicciones pendientes
        mlflow_uri: URI de MLflow
    
    Returns:
        Dict con métricas de evaluación
    """
    if predictions_df.empty:
        return {}
    
    # Descargar datos históricos para obtener retornos reales
    import yfinance as yf
    
    min_date = predictions_df['prediction_date'].min()
    max_date = predictions_df['evaluation_date'].max()
    
    # Descargar datos con buffer
    start_date = min_date - timedelta(days=10)
    end_date = max_date + timedelta(days=10)
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        print(f"⚠️  No se pudieron descargar datos para {ticker}")
        return {}
    
    # Calcular retornos reales
    evaluated = []
    
    for _, pred in predictions_df.iterrows():
        pred_date = pred['prediction_date']
        eval_date = pred['evaluation_date']
        
        # Obtener precio en fecha de predicción y evaluación
        try:
            price_pred = data.loc[pred_date:pred_date + timedelta(days=5), 'Close'].iloc[0]
            price_eval = data.loc[eval_date:eval_date + timedelta(days=5), 'Close'].iloc[0]
            
            actual_return = (price_eval - price_pred) / price_pred
            
            evaluated.append({
                'run_id': pred['run_id'],
                'prediction_date': pred_date,
                'evaluation_date': eval_date,
                'predicted_return': pred['predicted_return'],
                'actual_return': actual_return,
                'error': abs(actual_return - pred['predicted_return']),
                'direction_correct': np.sign(actual_return) == np.sign(pred['predicted_return'])
            })
        except (IndexError, KeyError):
            print(f"⚠️  No hay datos para {pred_date} -> {eval_date}")
            continue
    
    if not evaluated:
        return {}
    
    eval_df = pd.DataFrame(evaluated)
    
    # Calcular métricas
    metrics = {
        'mae': eval_df['error'].mean(),
        'rmse': np.sqrt((eval_df['error'] ** 2).mean()),
        'directional_accuracy': eval_df['direction_correct'].mean(),
        'ic': eval_df[['predicted_return', 'actual_return']].corr('spearman').iloc[0, 1],
        'n_evaluated': len(eval_df)
    }
    
    # Guardar evaluación en MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("E1_Simple_Production_Tracking")
    
    with mlflow.start_run(run_name=f"evaluation_{ticker}_{datetime.now().strftime('%Y%m%d')}"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("evaluation_type", "production_tracking")
        mlflow.log_param("n_predictions", len(eval_df))
        
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                mlflow.log_metric(k, float(v))
        
        # Guardar detalles como artifact
        eval_path = Path(f"/tmp/{ticker}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        eval_df.to_csv(eval_path, index=False)
        mlflow.log_artifact(str(eval_path), artifact_path="evaluations")
        eval_path.unlink()
        
        print(f"\n✓ Evaluación guardada en MLflow")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  IC: {metrics['ic']:.4f}")
        print(f"  Dir Acc: {metrics['directional_accuracy']:.2%}")
    
    # Actualizar status de predicciones a "evaluated"
    for run_id in eval_df['run_id']:
        mlflow.tracking.MlflowClient().set_tag(run_id, "status", "evaluated")
    
    return metrics


def compare_with_training_metrics(
    ticker: str,
    current_metrics: dict,
    model_run_id: str,
    mlflow_uri: str
) -> dict:
    """
    Compara métricas actuales con las del entrenamiento original.
    
    Returns:
        Dict con comparación y alertas
    """
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Obtener métricas del modelo original
    try:
        original_run = mlflow.get_run(model_run_id)
        original_metrics = {
            'mae': original_run.data.metrics.get('mae', None),
            'ic': original_run.data.metrics.get('ic', None),
            'directional_accuracy': original_run.data.metrics.get('directional_accuracy', None),
            'bt_sharpe': original_run.data.metrics.get('bt_sharpe', None)
        }
    except:
        print(f"⚠️  No se pudo obtener run original: {model_run_id}")
        return {}
    
    comparison = {}
    alerts = []
    
    for metric in ['mae', 'ic', 'directional_accuracy']:
        if metric in current_metrics and original_metrics.get(metric) is not None:
            current = current_metrics[metric]
            original = original_metrics[metric]
            
            # Calcular degradación
            if metric == 'mae':
                # MAE: menor es mejor
                degradation = (current - original) / original
                threshold = 0.20  # 20% peor
            else:
                # IC, Dir Acc: mayor es mejor
                degradation = (original - current) / original
                threshold = 0.15  # 15% peor
            
            comparison[metric] = {
                'original': original,
                'current': current,
                'degradation_pct': degradation * 100
            }
            
            if degradation > threshold:
                alerts.append(f"⚠️  {metric.upper()} degradó {degradation*100:.1f}% (threshold: {threshold*100:.0f}%)")
    
    return {'comparison': comparison, 'alerts': alerts}


def main():
    parser = argparse.ArgumentParser(description="Evaluación continua E1 Simple")
    parser.add_argument("--ticker", type=str, help="Ticker a evaluar")
    parser.add_argument("--evaluate-pending", action="store_true", help="Evaluar predicciones pendientes")
    parser.add_argument("--mlflow-docker", action="store_true", help="Usar MLflow en Docker")
    
    args = parser.parse_args()
    
    # Configurar MLflow
    mlflow_uri = get_mlflow_tracking_uri(args.mlflow_docker)
    
    print("=" * 80)
    print("E1 Simple - Evaluación Continua con MLflow")
    print("=" * 80)
    print(f"MLflow URI: {mlflow_uri}")
    print()
    
    if args.evaluate_pending:
        if not args.ticker:
            print("❌ Especifica --ticker para evaluar predicciones pendientes")
            sys.exit(1)
        
        print(f"📊 Buscando predicciones pendientes para {args.ticker}...")
        pending = get_pending_predictions(args.ticker, mlflow_uri)
        
        if pending.empty:
            print(f"✓ No hay predicciones pendientes para evaluar ({args.ticker})")
        else:
            print(f"✓ Encontradas {len(pending)} predicciones para evaluar")
            print()
            
            metrics = evaluate_predictions(args.ticker, pending, mlflow_uri)
            
            if metrics:
                # Comparar con entrenamiento original
                model_run_id = pending.iloc[0]['model_run_id']
                comparison = compare_with_training_metrics(
                    args.ticker, metrics, model_run_id, mlflow_uri
                )
                
                if comparison and comparison.get('alerts'):
                    print("\n" + "=" * 80)
                    print("🚨 ALERTAS DE DEGRADACIÓN")
                    print("=" * 80)
                    for alert in comparison['alerts']:
                        print(alert)
    else:
        print("Uso:")
        print("  1. Evaluar predicciones pendientes:")
        print("     python scripts/e1_continuous_evaluation.py --ticker AAPL --evaluate-pending")
        print()
        print("  2. Integrar en script de trading diario (guardar predicción):")
        print("     Ver función save_daily_prediction()")


if __name__ == "__main__":
    main()
