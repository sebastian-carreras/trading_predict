#!/usr/bin/env python3
"""
Script para inspeccionar y analizar la base de datos de Optuna (optuna_studies.db).

Proporciona consultas útiles para entender el progreso de la optimización
sin usar la UI de optuna-dashboard.
"""

import sqlite3
from pathlib import Path
from tabulate import tabulate
import pandas as pd


class OptunaDBAnalyzer:
    """Analizador de base de datos SQLite de Optuna."""
    
    def __init__(self, db_path: str = "optuna_studies.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def execute_query(self, query: str) -> list:
        """Ejecuta una query y retorna resultados."""
        cursor = self.conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    
    def get_all_studies(self) -> pd.DataFrame:
        """Retorna lista de todos los estudios con estadísticas."""
        query = """
        SELECT 
            s.study_name,
            COUNT(DISTINCT t.trial_id) as n_trials,
            ROUND(AVG(tv.value), 4) as avg_objective,
            ROUND(MAX(tv.value), 4) as best_objective,
            ROUND(MIN(tv.value), 4) as worst_objective,
            COUNT(CASE WHEN t.state = 'COMPLETE' THEN 1 END) as completed_trials,
            COUNT(CASE WHEN t.state = 'FAILED' THEN 1 END) as failed_trials
        FROM studies s
        LEFT JOIN trials t ON s.study_id = t.study_id
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        GROUP BY s.study_name
        ORDER BY best_objective DESC
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_best_trial(self, study_name: str) -> dict:
        """Retorna el mejor trial de un estudio específico."""
        query = """
        SELECT 
            t.trial_id,
            t.number,
            tv.value as objective,
            t.state,
            t.datetime_start,
            t.datetime_complete
        FROM studies s
        JOIN trials t ON s.study_id = t.study_id
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE s.study_name = ?
        ORDER BY tv.value DESC
        LIMIT 1
        """
        cursor = self.conn.cursor()
        cursor.execute(query, (study_name,))
        return dict(cursor.fetchone())
    
    def get_best_params(self, study_name: str) -> dict:
        """Retorna los parámetros del mejor trial de un estudio."""
        # Obtener trial_id del mejor trial primero
        best_trial_query = """
        SELECT t.trial_id
        FROM studies s
        JOIN trials t ON s.study_id = t.study_id
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE s.study_name = ?
        ORDER BY tv.value DESC
        LIMIT 1
        """
        cursor = self.conn.cursor()
        cursor.execute(best_trial_query, (study_name,))
        result = cursor.fetchone()
        if not result:
            return {}
        
        best_trial_id = result[0]
        
        params_query = """
        SELECT param_name, param_value
        FROM trial_params
        WHERE trial_id = ?
        """
        cursor.execute(params_query, (best_trial_id,))
        return {row[0]: row[1] for row in cursor.fetchall()}
    
    def get_trial_history(self, study_name: str, limit: int = 10) -> pd.DataFrame:
        """Retorna histórico de trials de un estudio (últimos N)."""
        query = """
        SELECT 
            t.number,
            tv.value as objective,
            t.state,
            t.datetime_start,
            ROUND(
                CAST((julianday(t.datetime_complete) - julianday(t.datetime_start)) * 24 * 60 AS FLOAT),
                2
            ) as duration_minutes
        FROM studies s
        JOIN trials t ON s.study_id = t.study_id
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE s.study_name = ?
        ORDER BY t.number DESC
        LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=(study_name, limit))
    
    def get_parameter_importance(self, study_name: str) -> pd.DataFrame:
        """Analiza qué parámetros son más importantes (correlation con objetivo)."""
        query = """
        SELECT 
            tp.param_name,
            COUNT(*) as n_occurrences,
            ROUND(AVG(CAST(tp.param_value AS FLOAT)), 4) as avg_value
        FROM studies s
        JOIN trials t ON s.study_id = t.study_id
        JOIN trial_params tp ON t.trial_id = tp.trial_id
        WHERE s.study_name = ? AND tp.param_name IS NOT NULL
        GROUP BY tp.param_name
        ORDER BY n_occurrences DESC
        """
        return pd.read_sql_query(query, self.conn, params=(study_name,))
    
    def export_to_csv(self, study_name: str, output_path: str = None):
        """Exporta todos los trials de un estudio a CSV."""
        if output_path is None:
            output_path = f"{study_name}_trials.csv"
        
        query = """
        SELECT 
            t.number,
            tv.value as objective,
            t.state,
            t.datetime_start,
            t.datetime_complete
        FROM studies s
        JOIN trials t ON s.study_id = t.study_id
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE s.study_name = ?
        ORDER BY t.number
        """
        
        df = pd.read_sql_query(query, self.conn, params=(study_name,))
        df.to_csv(output_path, index=False)
        print(f"✓ Exportado a: {output_path}")
        return df


def main():
    """Interfaz interactiva para analizar la DB."""
    analyzer = OptunaDBAnalyzer()
    
    print("="*80)
    print("ANALIZADOR DE OPTUNA STUDIES DATABASE")
    print("="*80)
    
    # 1. Resumen general
    print("\n1. RESUMEN GENERAL DE ESTUDIOS\n")
    studies_df = analyzer.get_all_studies()
    print(tabulate(studies_df, headers='keys', tablefmt='grid', showindex=False))
    
    # 2. Top 5 mejores estudios
    print("\n\n2. TOP 5 ESTUDIOS POR MEJOR OBJETIVO\n")
    top_studies = studies_df.head(5)
    for idx, row in top_studies.iterrows():
        print(f"  🏆 {row['study_name']}")
        print(f"     - Mejor objetivo: {row['best_objective']}")
        print(f"     - Promedio: {row['avg_objective']}")
        print(f"     - Trials completados: {row['completed_trials']}/{row['n_trials']}")
    
    # 3. Detalles del mejor estudio
    if len(studies_df) > 0:
        best_study_name = studies_df.iloc[0]['study_name']
        
        print(f"\n\n3. DETALLES DEL MEJOR ESTUDIO: {best_study_name}\n")
        
        best_trial = analyzer.get_best_trial(best_study_name)
        print(f"  Trial #{best_trial['number']} (ID: {best_trial['trial_id']})")
        print(f"  - Objetivo: {best_trial['objective']:.6f}")
        print(f"  - Estado: {best_trial['state']}")
        print(f"  - Inicio: {best_trial['datetime_start']}")
        
        best_params = analyzer.get_best_params(best_study_name)
        if best_params:
            print(f"\n  Mejores parámetros:")
            for param_name, param_value in best_params.items():
                print(f"    - {param_name}: {param_value}")
        
        # Histórico reciente
        print(f"\n  Últimos 5 trials:")
        history_df = analyzer.get_trial_history(best_study_name, limit=5)
        print(tabulate(history_df, headers='keys', tablefmt='simple', showindex=False))
    
    print("\n" + "="*80)
    print("Funciones disponibles:")
    print("  analyzer.get_all_studies()           → Resumen de todos los estudios")
    print("  analyzer.get_best_trial('study_name')  → Mejor trial de un estudio")
    print("  analyzer.get_best_params('study_name') → Parámetros del mejor trial")
    print("  analyzer.get_trial_history('study_name', limit=10) → Histórico")
    print("  analyzer.get_parameter_importance('study_name') → Importancia de params")
    print("  analyzer.export_to_csv('study_name') → Exportar a CSV")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
