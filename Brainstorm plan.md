- Hacer un analisis de objetivos largo, mediano y corto plazo del proyecto.


### Semana libre plan de trabajo
- [_] Plantear nomenglaturas para los modelos:
	- Baseline: referencia fija para comparar, no tiene por qué estar en producción.
	- Candidate: cualquier modelo nuevo que estás probando offline contra el baseline.
	- Challenger: modelo candidato que compite contra el modelo “oficial” usando los mismos datos (A/B, shadow, etc.).
	- Champion: modelo actualmente aprobado y desplegado en producción.
	- Retired / Archived: modelos antiguos guardados solo para histórico y auditoría.

- Hacer que el baseline model quede fijo aunque se modifiquen las features del modelo o cualquier otro parametro que se use en otros modelos.

- [_] Check the continuos evaluation "branch" that was never followed.

- [_] Estudiar walk-forward validation.


​


