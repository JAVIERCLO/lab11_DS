# Laboratorio 11. MLOps

CC3066 Data Science, Universidad del Valle de Guatemala, Semestre II 2026.
Integrantes: Javier Chávez, Genser Catalán y Juan Cruz.

Flujo reproducible para el Observatorio de Turismo Sostenible. Lee la serie mensual de
viajeros de 2009 a junio de 2026, la valida con seis controles de calidad, estima los
próximos tres meses con un método estacional ingenuo, lo evalúa contra una referencia,
registra cada ejecución en MLflow y publica solo las versiones aprobadas en un tablero.

## Cómo correr el proyecto

Requisitos previos: Docker Desktop con Docker Compose y Git.

**1. Clonar el repositorio y entrar a la carpeta**

```bash
git clone https://github.com/JAVIERCLO/lab11_DS.git && cd lab11_DS
```

**2. Construir la imagen y levantar MLflow y el tablero**

```bash
docker compose up -d --build
```

**3. Ejecutar el pipeline**

```bash
docker compose run --rm pipeline
```

**4. Abrir el tablero y MLflow**

El tablero queda en http://localhost:8501 y MLflow en http://localhost:5001, dentro del experimento `lab11-viajeros-guatemala`.

**5. Correr las pruebas**

```bash
docker compose run --rm pipeline python -m pytest -q
```

**6. Detener los servicios**

```bash
docker compose down
```

## Prueba de fallo y restauración

Genera copias sintéticas con problemas en `outputs/pruebas` y corre el pipeline con una de ellas. Debe fallar el control de unicidad y la versión aprobada sigue en el tablero.

```bash
docker compose run --rm pipeline python -m src.generar_pruebas
```

```bash
docker compose run --rm pipeline python -m src.pipeline --fuente outputs/pruebas/duplicado.csv
```

Para ver las versiones aprobadas y restaurar una anterior:

```bash
docker compose run --rm pipeline python -m src.restaurar
```

```bash
docker compose run --rm pipeline python -m src.restaurar 20261102T002728Z-2e47
```

## Notas

**Datos.** La fuente se lee de `data/serie_total_mensual.csv`, que se monta en los contenedores en solo lectura y no entra en la imagen. Para usar otra carpeta con los datos institucionales se define `DATA_DIR=/ruta/a/los/datos` antes de `docker compose`.

**Configuración.** Rutas, periodos, parámetros, factor de reparto y umbrales de aceptación están en `config/config.json`.

**Resultados.** Cada ejecución deja su reporte y su registro en `outputs/ejecuciones/<run_id>`. Las versiones aprobadas quedan en `outputs/publicado/versiones` y el tablero lee la que indica `outputs/publicado/vigente.json`. El historial de monitoreo está en `outputs/monitoreo`.

**MLflow.** La base y los artefactos se guardan en `mlflow/`, así que persisten aunque se borren los contenedores. El puerto 5000 lo usa AirPlay en macOS, por eso MLflow se expone en el 5001.

**Códigos de salida.** El pipeline devuelve 0 si publicó, 2 si la versión fue rechazada por la regla de aceptación y 1 si se detuvo por un control crítico.
