# Laboratorio 11 - Avance MLOps

Pipeline inicial para serie mensual de viajeros. **Propuesta provisional:** metodo ingenuo estacional (mismo mes del anio anterior), no afirmamos que sea uno de los modelos previos. Reemplazar por el procedimiento efectivamente utilizado por el equipo antes de la entrega final. La fuente `serie_total_mensual.csv` fue suministrada por el equipo; confirmar que su version es la fuente institucional autorizada. El factor de reparto es provisional (1.0) y debe confirmarse metodologicamente.

## Ejecucion

```bash
pip install -r requirements.txt
python src/pipeline.py
```

La salida se guarda en `outputs/`. `approved.csv` solo se crea o reemplaza si pasan todos los controles y la regla de aceptacion. Se conserva un respaldo de la version anterior. La regla MAE <= baseline MAE es provisional y debe justificarse antes de fijarla.

## Prueba de fallo controlado

Crear una copia con una fila duplicada y ejecutar `python src/pipeline.py --source data/serie_con_duplicado.csv`. Debe fallar `unicidad`, generar reporte JSON y no modificar `approved.csv`.

## Pendientes para entrega final

Integrar MLflow, Docker, monitoreo y restauracion documentada; confirmar destino, periodos, factor de reparto, modelo real y regla de aceptacion; conectar `approved.csv` al dashboard. Los valores de la fuente son decimales: revisar si representan conteos ajustados y documentar la unidad correcta.
