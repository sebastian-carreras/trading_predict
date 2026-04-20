# 📝 Reglas para escribir comentarios en este proyecto

1. **Explica el POR QUÉ, nunca el QUÉ.**
   No describas lo que hace el código; explica la razón detrás de cada decisión.
   Si el código es legible por sí solo, no añadas comentario.

2. **No comentes lo obvio.**
   Si el nombre de la función o variable ya lo dice, omite el comentario.
   Ejemplo de lo que NO hacer: `# Suma a + b` sobre `return a + b`.

3. **Usa Google Docstrings en todas las funciones públicas.**
   Incluye siempre: descripción de una línea, Args, Returns y Raises si aplica.
   Las funciones privadas (_nombre) solo necesitan comentario si su lógica es compleja.

4. **Usa etiquetas estándar para código pendiente o riesgoso.**
   - `# TODO:` → tarea pendiente con detalle de qué y por qué
   - `# FIXME:` → bug conocido con descripción del problema
   - `# WARNING:` → comportamiento no obvio que puede romper algo
   - `# HACK:` → solución temporal, explica qué la hace temporal

5. **Documenta los edge cases y restricciones de inputs.**
   Si un parámetro tiene un rango válido, una unidad, o un formato esperado,
   indícalo en el comentario. Ejemplo: `# pct: float entre 0.0 y 1.0`

6. **Agrega un bloque de contexto al inicio de cada módulo nuevo.**
   Incluye: propósito del módulo, dependencias clave y qué NO debe hacerse aquí.
   Ejemplo:
   # CONTEXT: Feature engineering para modelos GRU.
   # Depende de: pandas-ta, yfinance. NO usar sklearn en este módulo.

7. **Escribe los comentarios para un desarrollador Python con nivel intermedio.**
   No expliques sintaxis básica. Sí explica lógica de negocio, fórmulas matemáticas
   y decisiones de arquitectura que no sean inmediatamente evidentes.

8. **Mantén los comentarios en el mismo idioma que el código base.**

9. **Si un comentario existente ya no refleja el código actual, márcalo.**
   Usa `# OUTDATED:` al inicio del comentario desactualizado en lugar de borrarlo,
   para que el desarrollador decida si actualizarlo o eliminarlo.

10. **Un comentario inline por bloque lógico, no por línea.**
    Agrupa líneas relacionadas en un bloque y ponle UN comentario encima.
    Nunca pongas comentarios al final de cada línea de un bloque; eso genera ruido.