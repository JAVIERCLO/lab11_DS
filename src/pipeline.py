import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src import controles, modelo, monitoreo, publicacion

RAIZ = Path(__file__).resolve().parents[1]


class EjecucionDetenida(Exception):
    def __init__(self, etapa, causa):
        super().__init__(causa)
        self.etapa = etapa


def cargar_config(ruta):
    return json.loads(Path(ruta).read_text())


def leer_commit(raiz):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True, text=True, check=True).stdout.strip()
        cambios = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=raiz, capture_output=True, text=True, check=True).stdout.strip()
        return commit, bool(cambios)
    except (OSError, subprocess.CalledProcessError):
        pass
    # El contenedor no trae git, se lee la referencia directo de .git montado en solo lectura
    cabeza = raiz / ".git" / "HEAD"
    if not cabeza.exists():
        return os.environ.get("GIT_COMMIT", "desconocido"), None
    referencia = cabeza.read_text().strip()
    if not referencia.startswith("ref: "):
        return referencia, None
    nombre = referencia[5:]
    archivo = raiz / ".git" / nombre
    if archivo.exists():
        return archivo.read_text().strip(), None
    empacadas = raiz / ".git" / "packed-refs"
    if empacadas.exists():
        for linea in empacadas.read_text().splitlines():
            if linea.endswith(" " + nombre):
                return linea.split()[0], None
    return "desconocido", None


def huella_estimaciones(estimaciones):
    contenido = estimaciones[["destino", "periodo_estimado", "valor", "unidad"]].to_csv(index=False)
    return hashlib.sha256(contenido.encode()).hexdigest()


def configurar_mlflow(cfg, raiz):
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        uri = cfg["tracking_uri_local"]
        if uri.startswith("sqlite:///"):
            ruta_db = raiz / uri.replace("sqlite:///", "")
            ruta_db.parent.mkdir(parents=True, exist_ok=True)
            uri = f"sqlite:///{ruta_db}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["experimento_mlflow"])
    return uri


def versiones_dependencias():
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "mlflow": mlflow.__version__,
    }


def ejecutar(ruta_config, fuente=None, salidas=None, raiz=RAIZ):
    cfg = cargar_config(ruta_config)
    if fuente:
        cfg["fuente"]["ruta"] = str(fuente)
    salidas = Path(salidas) if salidas else raiz / cfg["rutas"]["salidas"]
    semilla = cfg["procedimiento"]["semilla"]
    random.seed(semilla)
    np.random.seed(semilla)

    ahora = datetime.now(timezone.utc)
    run_id = f"{ahora:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:4]}"
    fecha_generacion = ahora.isoformat(timespec="seconds")
    registro = monitoreo.RegistroEjecucion(salidas, run_id)
    carpeta = registro.carpeta
    ruta_fuente = Path(cfg["fuente"]["ruta"])
    ruta_fuente = ruta_fuente if ruta_fuente.is_absolute() else raiz / ruta_fuente
    commit, sucio = leer_commit(raiz)

    configurar_mlflow(cfg, raiz)
    corrida = mlflow.start_run(run_name=run_id)
    periodos = cfg["periodos"]
    mlflow.set_tags({
        "run_id": run_id,
        "fecha_ejecucion": fecha_generacion,
        "git_commit": commit,
        "git_cambios_sin_commit": str(sucio),
        "destino": cfg["destino"],
        "fuente": str(ruta_fuente.relative_to(raiz)) if ruta_fuente.is_relative_to(raiz) else str(ruta_fuente),
    })
    mlflow.log_params({
        "ajuste_fin": periodos["ajuste_fin"],
        "evaluacion_inicio": periodos["evaluacion_inicio"],
        "evaluacion_fin": periodos["evaluacion_fin"],
        "cobertura_inicio": periodos["cobertura_inicio"],
        "cobertura_fin": periodos["cobertura_fin"],
        **{k: v for k, v in cfg["procedimiento"].items()},
        **{f"aceptacion_{k}": v for k, v in cfg["aceptacion"].items()},
        "variacion_maxima_monitoreo": cfg["monitoreo"]["variacion_maxima"],
        **{f"version_{k}": v for k, v in versiones_dependencias().items()},
    })
    mlflow.log_dict(cfg, "config.json")

    reporte = {"run_id": run_id, "mlflow_run_id": corrida.info.run_id, "git_commit": commit, "fecha": fecha_generacion}
    etapa = "lectura"
    estado = "FALLIDA"
    publicacion_resultado = "No publicada, se conserva la versión aprobada vigente"
    try:
        if not ruta_fuente.exists():
            raise EjecucionDetenida(etapa, f"No existe la fuente {ruta_fuente}")
        datos = pd.read_csv(ruta_fuente)
        reporte["fuente_sha256"] = publicacion.huella(ruta_fuente)
        mlflow.set_tag("fuente_sha256", reporte["fuente_sha256"])
        registro.etapa(etapa)

        etapa = "validacion"
        resultados = controles.validar(datos, cfg)
        reporte["controles"] = resultados
        mlflow.log_dict(resultados, "controles.json")
        for r in resultados:
            mlflow.set_tag(f"control_{r['control']}", r["estado"])
            if r["estado"] == controles.ADVERTENCIA:
                registro.advertencia(f"{r['control']}: {r['detalle']}")
        resumen = controles.resumen_calidad(datos, cfg)
        reporte["resumen_calidad"] = resumen
        for alerta in monitoreo.verificar_calidad(resumen, monitoreo.ultimo_resumen_exitoso(salidas)):
            registro.alerta("calidad", alerta)
        if controles.hay_fallas(resultados):
            fallidos = [f"{r['control']}: {r['detalle']}" for r in resultados if r["estado"] == controles.FALLA]
            raise EjecucionDetenida(etapa, "Controles críticos fallidos. " + " | ".join(fallidos))
        registro.etapa(etapa)

        etapa = "preparacion"
        serie = controles.preparar_serie(datos, cfg)
        registro.etapa(etapa)

        etapa = "ajuste"
        parametros = modelo.parametros_ajustados(serie, cfg)
        mlflow.log_dict(parametros, "parametros_modelo.json")
        registro.etapa(etapa)

        etapa = "evaluacion"
        validacion = modelo.validacion_temporal(serie, cfg)
        if (validacion["ultimo_mes_usado"] >= validacion["periodo"]).any():
            raise EjecucionDetenida(etapa, "Fuga temporal: se usó información del mes evaluado o posterior")
        evaluacion = modelo.evaluar(validacion)
        reporte["evaluacion"] = evaluacion
        mlflow.log_metrics(evaluacion)
        validacion.to_csv(carpeta / "validacion_temporal.csv", index=False)
        mlflow.log_artifact(carpeta / "validacion_temporal.csv")
        registro.etapa(etapa)

        etapa = "estimacion"
        estimaciones = modelo.generar_estimaciones(serie, parametros, cfg, run_id, fecha_generacion)
        registro.etapa(etapa)

        etapa = "almacenamiento"
        ruta_candidatas = carpeta / "estimaciones_candidatas.csv"
        estimaciones.to_csv(ruta_candidatas, index=False)
        mlflow.log_artifact(ruta_candidatas)
        reporte["huella_estimaciones"] = huella_estimaciones(estimaciones)
        mlflow.set_tag("huella_estimaciones", reporte["huella_estimaciones"])
        registro.etapa(etapa)

        etapa = "monitoreo"
        anteriores, vigente = publicacion.leer_estimaciones_vigentes(salidas)
        for alerta in monitoreo.verificar_actualizacion(serie, estimaciones, cfg):
            registro.alerta("actualizacion", alerta)
        for alerta in monitoreo.verificar_resultados(estimaciones, serie, anteriores, cfg):
            registro.alerta("resultados", alerta)
        seguimiento = monitoreo.error_de_seguimiento(anteriores, serie)
        reporte["error_seguimiento"] = seguimiento
        if seguimiento:
            mlflow.log_metric("mae_seguimiento", seguimiento["mae"])
        registro.etapa(etapa)

        etapa = "aceptacion"
        problemas = modelo.integridad_salida(estimaciones, cfg)
        aceptada, criterios = modelo.decidir_aceptacion(True, problemas, evaluacion, cfg)
        reporte["aceptacion"] = {"aceptada": aceptada, "criterios": criterios, "problemas_salida": problemas}
        mlflow.log_dict(reporte["aceptacion"], "aceptacion.json")
        mlflow.set_tag("aceptada", str(aceptada))
        registro.etapa(etapa)

        etapa = "publicacion"
        if aceptada:
            metadatos = {
                "run_id": run_id,
                "mlflow_run_id": corrida.info.run_id,
                "git_commit": commit,
                "fecha_generacion": fecha_generacion,
                "destino": cfg["destino"],
                "unidad": cfg["procedimiento"]["unidad"],
                "version_procedimiento": f"{cfg['procedimiento']['metodo']}-{cfg['procedimiento']['version']}",
                "fuente_sha256": reporte["fuente_sha256"],
                "ultimo_mes_observado": parametros["ultimo_mes_observado"],
                "periodo_evaluado": [f"{validacion['periodo'].min():%Y-%m}", f"{validacion['periodo'].max():%Y-%m}"],
                "metricas": evaluacion,
            }
            publicacion.publicar(salidas, run_id, estimaciones, metadatos)
            monitoreo.guardar_resumen_calidad(salidas, run_id, resumen)
            estado = "PUBLICADA"
            publicacion_resultado = f"Publicada como versión vigente {run_id}"
        else:
            estado = "RECHAZADA"
            rechazos = [k for k, v in criterios.items() if not v]
            publicacion_resultado = f"Rechazada por {rechazos}, se conserva la versión aprobada vigente"
            registro.advertencia(publicacion_resultado)
        registro.etapa(etapa)
    except EjecucionDetenida as error:
        registro.error(f"Ejecución detenida en {error.etapa}: {error}")
        reporte["error"] = {"etapa": error.etapa, "causa": str(error)}
    except Exception as error:
        registro.error(f"Error inesperado en {etapa}: {error!r}")
        reporte["error"] = {"etapa": etapa, "causa": repr(error)}

    reporte["estado"] = estado
    reporte["publicacion"] = publicacion_resultado
    vigente = publicacion.leer_vigente(salidas)
    reporte["version_vigente"] = vigente["run_id"] if vigente else None
    if vigente is None:
        registro.advertencia("No existe una versión aprobada, el tablero indicará que no hay estimaciones disponibles")
    registro.cerrar(publicacion_resultado)
    reporte["monitoreo"] = registro.datos
    (carpeta / "reporte.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False, default=str))

    mlflow.set_tags({"estado": estado, "publicacion": publicacion_resultado})
    if "error" in reporte:
        mlflow.set_tags({"etapa_fallida": reporte["error"]["etapa"], "causa_fallo": reporte["error"]["causa"][:500]})
    mlflow.log_artifact(carpeta / "reporte.json")
    mlflow.log_artifact(carpeta / "registro.log")
    mlflow.end_run(status="FAILED" if estado == "FALLIDA" else "FINISHED")
    return reporte


def main():
    parser = argparse.ArgumentParser(description="Pipeline de estimación mensual de viajeros")
    parser.add_argument("--config", default=str(RAIZ / "config" / "config.json"))
    parser.add_argument("--fuente", default=None, help="Ruta alternativa de la fuente, útil para pruebas de fallo")
    parser.add_argument("--salidas", default=None)
    args = parser.parse_args()
    reporte = ejecutar(args.config, args.fuente, args.salidas)
    print(json.dumps({k: reporte.get(k) for k in ["run_id", "estado", "publicacion", "version_vigente", "evaluacion"]}, indent=2, ensure_ascii=False, default=str))
    return {"PUBLICADA": 0, "RECHAZADA": 2}.get(reporte["estado"], 1)


if __name__ == "__main__":
    sys.exit(main())
