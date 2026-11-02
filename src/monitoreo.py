import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RegistroEjecucion:
    def __init__(self, salidas, run_id):
        self.salidas = Path(salidas)
        self.run_id = run_id
        self.carpeta = self.salidas / "ejecuciones" / run_id
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.datos = {
            "run_id": run_id,
            "inicio": ahora(),
            "fin": None,
            "etapas_completadas": [],
            "advertencias": [],
            "errores": [],
            "alertas": [],
            "publicacion": None,
        }
        self.lineas = []

    def anotar(self, nivel, mensaje):
        linea = f"{ahora()} [{nivel}] {mensaje}"
        self.lineas.append(linea)
        print(linea, flush=True)

    def etapa(self, nombre):
        self.datos["etapas_completadas"].append(nombre)
        self.anotar("INFO", f"Etapa completada: {nombre}")

    def advertencia(self, mensaje):
        self.datos["advertencias"].append(mensaje)
        self.anotar("ADVERTENCIA", mensaje)

    def error(self, mensaje):
        self.datos["errores"].append(mensaje)
        self.anotar("ERROR", mensaje)

    def alerta(self, verificacion, mensaje):
        self.datos["alertas"].append({"verificacion": verificacion, "mensaje": mensaje})
        self.anotar("ALERTA", f"{verificacion}: {mensaje}")

    def cerrar(self, publicacion):
        self.datos["fin"] = ahora()
        self.datos["publicacion"] = publicacion
        self.anotar("INFO", f"Resultado de la publicación: {publicacion}")
        (self.carpeta / "registro.log").write_text("\n".join(self.lineas) + "\n")
        carpeta_monitoreo = self.salidas / "monitoreo"
        carpeta_monitoreo.mkdir(parents=True, exist_ok=True)
        with open(carpeta_monitoreo / "ejecuciones.jsonl", "a") as archivo:
            archivo.write(json.dumps(self.datos, ensure_ascii=False) + "\n")


def ultimo_resumen_exitoso(salidas):
    ruta = Path(salidas) / "monitoreo" / "calidad.jsonl"
    if not ruta.exists():
        return None
    lineas = [json.loads(l) for l in ruta.read_text().splitlines() if l.strip()]
    return lineas[-1] if lineas else None


def guardar_resumen_calidad(salidas, run_id, resumen):
    ruta = Path(salidas) / "monitoreo" / "calidad.jsonl"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "a") as archivo:
        archivo.write(json.dumps({"run_id": run_id, **resumen}) + "\n")


def verificar_actualizacion(serie, estimaciones, cfg):
    alertas = []
    esperado = pd.Timestamp(cfg["periodos"]["cobertura_fin"])
    if serie.index.max() < esperado:
        alertas.append(f"La fuente llega a {serie.index.max():%Y-%m} y se esperaba {esperado:%Y-%m}")
    primer_estimado = pd.Timestamp(estimaciones["periodo_estimado"].min())
    if primer_estimado != serie.index.max() + pd.offsets.MonthBegin(1):
        alertas.append(f"Las estimaciones empiezan en {primer_estimado:%Y-%m} y no en el mes siguiente al último observado")
    return alertas


def verificar_calidad(resumen_actual, resumen_anterior):
    if resumen_anterior is None:
        return []
    if "meses_cubiertos" not in resumen_actual:
        return ["No se pudo medir la calidad porque faltan columnas indispensables"]
    alertas = []
    for clave in ["faltantes", "duplicados"]:
        if resumen_actual[clave] > resumen_anterior[clave]:
            alertas.append(f"{clave} subió de {resumen_anterior[clave]} a {resumen_actual[clave]}")
    if resumen_actual["meses_cubiertos"] < resumen_anterior["meses_cubiertos"]:
        alertas.append(f"La cobertura bajó de {resumen_anterior['meses_cubiertos']} a {resumen_actual['meses_cubiertos']} meses")
    if resumen_actual["filas"] != resumen_anterior["filas"]:
        alertas.append(f"La fuente pasó de {resumen_anterior['filas']} a {resumen_actual['filas']} filas")
    return alertas


def verificar_resultados(estimaciones, serie, estimaciones_anteriores, cfg):
    alertas = []
    umbral = cfg["monitoreo"]["variacion_maxima"]
    valores = estimaciones["valor"].to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        alertas.append("Hay estimaciones no finitas")
    if (valores < 0).any():
        alertas.append("Hay estimaciones negativas")
    for fila in estimaciones.itertuples():
        mismo_mes_anterior = pd.Timestamp(fila.periodo_estimado) - pd.DateOffset(years=1)
        if mismo_mes_anterior in serie.index:
            variacion = fila.valor / serie.loc[mismo_mes_anterior] - 1
            if abs(variacion) > umbral:
                alertas.append(f"{fila.periodo_estimado}: variación interanual de {variacion:.1%} supera el {umbral:.0%}")
    if estimaciones_anteriores is not None:
        anteriores = estimaciones_anteriores.set_index("periodo_estimado")["valor"]
        for fila in estimaciones.itertuples():
            if fila.periodo_estimado in anteriores.index:
                variacion = fila.valor / anteriores.loc[fila.periodo_estimado] - 1
                if abs(variacion) > umbral:
                    alertas.append(f"{fila.periodo_estimado}: cambia {variacion:.1%} frente a la versión aprobada anterior")
    return alertas


def error_de_seguimiento(estimaciones_anteriores, serie):
    if estimaciones_anteriores is None:
        return None
    anteriores = estimaciones_anteriores.assign(periodo=pd.to_datetime(estimaciones_anteriores["periodo_estimado"]))
    comparables = anteriores[anteriores["periodo"].isin(serie.index)]
    if comparables.empty:
        return None
    observados = serie.loc[comparables["periodo"]].to_numpy()
    return {
        "meses": len(comparables),
        "mae": float(np.mean(np.abs(observados - comparables["valor"].to_numpy()))),
    }
