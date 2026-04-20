# Heurísticas Priorizadas y Brechas de Implementación

## 1. Sección formal para la tesis

### 1.1 Heurísticas metodológicas y operativas priorizadas

Para mantener coherencia entre el objetivo académico del proyecto, la evidencia empírica del análisis exploratorio y la implementación efectiva de los pipelines de entrenamiento, se definieron un conjunto de heurísticas prioritarias. Estas heurísticas no deben interpretarse como reglas arbitrarias de ingeniería, sino como criterios de diseño orientados a maximizar validez metodológica, robustez fuera de muestra y utilidad práctica de las señales generadas.

La primera heurística consiste en modelar retornos forward en lugar de precios absolutos. En series financieras, el precio presenta no estacionariedad fuerte, cambios de escala y dependencia del nivel nominal del activo. En cambio, el retorno forward al horizonte relevante permite alinear la variable objetivo con la decisión real del sistema: estimar si existe un movimiento futuro con magnitud suficiente para justificar una recomendación de compra, mantenimiento o salida.

La segunda heurística establece que el embargo temporal debe ser, como mínimo, igual al horizonte de predicción. Dado que los targets forward generan solapamiento entre observaciones consecutivas, una separación insuficiente entre entrenamiento y prueba induce leakage temporal. Por lo tanto, cuando el modelo predice a 90 días, el embargo mínimo debe ser de 90 días; cuando predice a 20 días, el embargo mínimo debe ser de 20 días; y en intradía debe replicarse el mismo principio en barras.

La tercera heurística indica que las ventanas de features deben estar alineadas con la escala temporal de cada estrategia. Una estrategia de largo plazo requiere features que describan tendencia persistente, régimen de volatilidad y momentum de varias semanas o meses. En cambio, una estrategia de mediano plazo necesita indicadores más reactivos, y una estrategia intradiaria debe apoyarse en microestructura, volatilidad de corto plazo y estacionalidad horaria. Esta alineación evita introducir ruido innecesario y mejora la interpretabilidad del modelo.

La cuarta heurística es normalizar por ticker utilizando exclusivamente estadísticas estimadas en entrenamiento. El universo del proyecto contiene activos argentinos y estadounidenses con diferencias estructurales de retorno nominal, volatilidad y liquidez. Si la normalización se realiza de forma agregada o usando información futura, se contaminan las comparaciones entre mercados y se introduce sesgo en la validación. Por ello, la normalización debe ser por activo y con parámetros ajustados únicamente sobre la muestra de entrenamiento de cada fold.

La quinta heurística consiste en reducir multicolinealidad antes del entrenamiento. En contextos financieros es frecuente que varios indicadores representen esencialmente la misma señal con diferentes parametrizaciones. Mantener features altamente redundantes aumenta complejidad, dificulta la interpretación y puede degradar estabilidad fuera de muestra. En consecuencia, se privilegia un conjunto parsimonioso de variables con aporte informativo diferenciado.

La sexta heurística reconoce que los retornos financieros presentan colas pesadas, outliers y desviaciones sistemáticas respecto de la normalidad. Por esta razón, la elección de la función de pérdida y de la arquitectura del modelo debe ser robusta a observaciones extremas. En este proyecto, dicha heurística justifica el uso de Huber loss, regularización moderada y validación temporal estricta por encima de alternativas optimizadas exclusivamente para error cuadrático medio.

La séptima heurística define que una señal predictiva solo debe transformarse en recomendación operativa cuando supera un umbral económico mínimo. No alcanza con predecir un retorno positivo; ese retorno esperado debe compensar al menos costos de transacción, fricción operativa y margen de error del modelo. Esta regla es especialmente importante en intradía, donde la relación señal-ruido es muy baja y los costos relativos son más altos.

La octava heurística establece que la evaluación principal debe combinar métricas de machine learning y métricas de trading. Un modelo con bajo error de regresión puede ser inútil desde el punto de vista operativo si genera señales no rentables, con Sharpe negativo o drawdowns incompatibles con el perfil de riesgo buscado. Por ello, métricas como Information Coefficient, directional accuracy, Sharpe y Calmar deben integrarse explícitamente en la selección y promoción de modelos.

La novena heurística sostiene que el entrenamiento y el monitoreo deben realizarse por ticker. El análisis exploratorio muestra diferencias estructurales claras entre mercados, sectores y perfiles de activo. En consecuencia, asumir un comportamiento homogéneo entre todos los instrumentos del universo llevaría a mezclar regímenes incompatibles y a debilitar la señal específica de cada ticker.

La décima heurística, relevante para la estrategia intradiaria, es priorizar selectividad por encima de frecuencia. En presencia de costos relativamente elevados y retornos esperados de muy baja magnitud, la rentabilidad no surge de operar más, sino de operar mejor. Bajo esta lógica, conviene exigir señales más robustas, filtrar ruido y aceptar menor cantidad de trades si eso mejora profit factor, Sharpe y estabilidad del equity curve.

### 1.2 Traducción resumida por estrategia

En E1, estas heurísticas se traducen en un modelo conservador de horizonte largo, con features de tendencia y momentum de varias semanas, validación walk-forward con embargo de 90 días y énfasis en robustez fuera de muestra. En E2, las heurísticas llevan a un modelo más reactivo, con features de 1 a 20 días, rebalanceo más frecuente y mayor peso relativo de la precisión direccional. En E3, las mismas heurísticas implican que la selectividad operativa, la compatibilidad entre umbral y costo, y la validación temporal intradiaria son más importantes que la mejora marginal en métricas puramente regresivas.

## 2. Revisión de consistencia de implementación

### 2.1 Resumen ejecutivo

El proyecto implementa de forma sólida la mayoría de las heurísticas prioritarias en E1 y E2. En cambio, E3 presenta una implementación parcial: su diseño conceptual está bien orientado, pero todavía no traslada consistentemente varias heurísticas críticas al pipeline operativo principal.

### 2.2 Estado por heurística

| Heurística priorizada | Estado actual | Observación |
|---|---|---|
| Predecir retornos forward, no precios | Implementada | E1, E2 y E3 construyen targets forward sobre retornos logarítmicos. |
| Embargo >= horizonte | Parcial | E1 y E2 lo implementan; E3 todavía usa `time_split` simple en el pipeline principal. |
| Features alineadas con horizonte | Implementada | La construcción de features está bien diferenciada entre 90d, 20d y 30 min. |
| Normalización por ticker con stats de train | Implementada | E1, E2 y E3 normalizan usando solo entrenamiento. |
| Poda de multicolinealidad | Parcial | E1 y E2 sí; E3 todavía está más cerca de una versión base con revisión pendiente. |
| Robustez a outliers y colas pesadas | Implementada | Huber loss está presente en las tres estrategias. |
| Operar solo si la señal supera un umbral económico | Parcial | E1 y E2 están razonablemente alineadas; E3 tiene umbrales por debajo del costo round-trip. |
| Evaluar con métricas de trading además de ML | Implementada con matices | E1 y E2 lo hacen bien; en E3 hay desalineación entre comentario, scoring y pipeline. |
| Entrenar y monitorear por ticker | Implementada | El flujo está organizado por ticker y el lifecycle también. |
| En intradía, priorizar selectividad | Parcial | La intención está, pero todavía no se traduce en filtros operativos suficientemente estrictos. |

### 2.3 Brechas más importantes detectadas

#### Brecha 1. E3 no replica todavía la validación temporal robusta de E1/E2

La estrategia intradiaria sigue usando un `time_split` simple en el pipeline principal. Esto deja una asimetría metodológica respecto de E1 y E2, donde el walk-forward con embargo es parte central del diseño. Dado que E3 también trabaja con targets forward solapados, la ausencia de embargo intradiario debilita la defensa metodológica del resultado fuera de muestra.

**Impacto**: alto. Esta es la principal brecha metodológica de E3.

**Acción recomendada**: migrar E3 a un esquema walk-forward con embargo mínimo de 6 barras, idealmente configurable desde `base.yaml` de forma análoga a E1/E2.

#### Brecha 2. Los umbrales operativos de E3 son menores que el costo de transacción round-trip

En la configuración actual, E3 usa `tau_buy=0.001` y `tau_sell=0.001`, equivalentes a 10 bps en valor absoluto, mientras que el costo round-trip configurado es de 20 bps. Esto contradice la heurística de operar solo cuando la magnitud esperada supera un umbral económico mínimo.

**Impacto**: alto. Incluso con señal correcta, la operación puede no cubrir costos.

**Acción recomendada**: redefinir los umbrales intradiarios en función del costo total, por ejemplo usando un mínimo mayor que 20 bps o una regla dependiente de costo más margen de seguridad.

#### Brecha 3. La heurística declarada para promoción de E3 no coincide con el scoring configurado

En la configuración de lifecycle se indica que E3 debería dar mayor peso a `profit_factor` y `directional_accuracy`. Sin embargo, los pesos definidos utilizan `bt_sharpe`, `ml_ic`, `ml_directional_accuracy` y `bt_calmar`, sin incluir `profit_factor`. Esto genera una inconsistencia entre la heurística expresada y la regla efectiva de decisión.

**Impacto**: medio-alto. La documentación y la configuración no están alineadas.

**Acción recomendada**: decidir una sola política y reflejarla en el scoring real. Si `profit_factor` es importante para E3, debe entrar explícitamente al composite score o eliminarse de la descripción.

#### Brecha 4. E3 aún no materializa selectividad intradiaria como filtro operativo fuerte

El EDA intradiario concluye que solo una fracción pequeña del target supera el costo de transacción y que la estrategia debe ser altamente selectiva. Sin embargo, el pipeline actual combina las predicciones del ensemble por promedio simple y ejecuta señales por umbral, sin una capa explícita de consenso fuerte o filtrado adicional para reducir trades marginales.

**Impacto**: medio. No invalida el pipeline, pero limita la coherencia entre diagnóstico y ejecución.

**Acción recomendada**: evaluar filtros de consenso entre miembros del ensemble, thresholds dependientes de volatilidad o reglas de no-trade cuando la señal esperada queda cerca del costo.

#### Brecha 5. La heurística de reducción de redundancia está madura en E1/E2, pero no cerrada en E3

E1 y E2 ya traducen el EDA a decisiones concretas de poda de features. E3 todavía conserva un set compacto, pero sin la misma madurez de depuración y sin una decisión final tan clara sobre qué pares multicolineales deben eliminarse o mantenerse.

**Impacto**: medio-bajo. No es el principal problema hoy, pero sí una diferencia de madurez respecto de las estrategias diarias.

**Acción recomendada**: cerrar un criterio explícito para mantener o eliminar pares con correlación alta consistente en varios tickers.

### 2.4 Conclusión práctica

Desde el punto de vista de la tesis, E1 y E2 ya expresan bastante bien las heurísticas priorizadas en código. La mayor oportunidad de mejora está en E3. Si el objetivo es fortalecer la coherencia metodológica global del proyecto, el siguiente foco debería ser: primero validación temporal robusta en E3, segundo reparametrización de umbrales intradiarios respecto de costos, y tercero alineación entre scoring de promoción y criterio operativo real.
