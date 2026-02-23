# Contexto del proyecto

## Qué es este proyecto
Trabajo Final de la Carrera de Especialización en Inteligencia Artificial de FIUBA (programa de 1 año).
Autor: Ing. Sebastian Carreras. Director: Dr. Facundo Lucianna.
Período: octubre 2024 – octubre 2025. Presupuesto: ~600 horas.

## Objetivo
Construir un sistema de IA que apoye la toma de decisiones de compra y venta de activos financieros (acciones, bonos, monedas) usando modelos de deep learning, datos en tiempo real e interfaz de usuario.

## Alcance — qué está incluido
- Modelos de ML/DL para predicción de precios (GRU, LSTM, ensemble, pairs trading)
- Adquisición de datos históricos y en tiempo real via APIs financieras
- 3 estrategias de inversión con distintos perfiles de riesgo (alto, medio, bajo)
- Alertas de compra/venta y de posibles arbitrajes
- Dashboard para visualizar recomendaciones y métricas de riesgo
- Backtesting y evaluación de rendimiento

## Alcance — qué NO está incluido
- Ejecución automática de órdenes (no hay trading en vivo ni gestión de capital)
- Integración con sistemas de gestión de portafolio o brokers
- Asesoría financiera personalizada
- Soporte multiidioma (solo español)

## Restricciones clave a tener en cuenta
- Este es un **proyecto académico de 1 año**, no un sistema productivo
- La complejidad debe ser acorde al contexto académico: prototipo funcional + evaluación sólida
- Error máximo de predicción esperado: 20% en condiciones de mercado estables
- El sistema debe operar ~10 horas continuas (cubre horario de mercados argentino y estadounidense)
- Entregables: código fuente, documentación técnica, manual de usuario, informe final

## Al hacer recomendaciones
- Preferir soluciones simples y probadas por sobre alternativas complejas o de vanguardia
- Priorizar que los modelos funcionen de punta a punta antes de optimizar precisión marginal
- Mantener la infraestructura liviana (local + nube, sin sobreingeniería)
- La audiencia es el director de tesis y evaluadores académicos, no usuarios finales
