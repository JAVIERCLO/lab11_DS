import numpy as np
import pandas as pd


def estimar_mes(historia, fecha, rezago, factor=1.0):
    base = fecha - pd.DateOffset(months=rezago)
    if base not in historia.index:
        raise ValueError(f"No hay observación en {base:%Y-%m} para estimar {fecha:%Y-%m}")
    return float(historia.loc[base]) * factor


def referencia_mes(historia, factor=1.0):
    return float(historia.iloc[-1]) * factor


def meses_evaluacion(serie, cfg):
    periodos = cfg["periodos"]
    excluidos = pd.to_datetime(periodos["meses_sin_observacion_comparable"])
    tramo = serie.loc[periodos["evaluacion_inicio"]:periodos["evaluacion_fin"]]
    return tramo.index[~tramo.index.isin(excluidos)]


def validacion_temporal(serie, cfg):
    rezago = cfg["procedimiento"]["rezago_estacional"]
    filas = []
    for fecha in meses_evaluacion(serie, cfg):
        historia = serie.loc[serie.index < fecha]
        filas.append({
            "periodo": fecha,
            "observado": float(serie.loc[fecha]),
            "estimado": estimar_mes(historia, fecha, rezago),
            "referencia": referencia_mes(historia),
            "ultimo_mes_usado": historia.index.max(),
        })
    if not filas:
        raise ValueError("El periodo de evaluación no tiene meses observados comparables")
    return pd.DataFrame(filas)


def metricas(observado, estimado):
    error = np.asarray(observado) - np.asarray(estimado)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.mean(np.abs(error) / np.asarray(observado)) * 100),
    }


def evaluar(validacion):
    procedimiento = metricas(validacion["observado"], validacion["estimado"])
    referencia = metricas(validacion["observado"], validacion["referencia"])
    resultado = {f"{k}_procedimiento": v for k, v in procedimiento.items()}
    resultado.update({f"{k}_referencia": v for k, v in referencia.items()})
    resultado["razon_mae_referencia"] = procedimiento["mae"] / referencia["mae"]
    resultado["meses_evaluados"] = len(validacion)
    return resultado


def parametros_ajustados(serie, cfg):
    procedimiento = cfg["procedimiento"]
    rezago = procedimiento["rezago_estacional"]
    if procedimiento["meses_pronostico"] > rezago:
        raise ValueError("El horizonte no puede superar el rezago estacional sin usar estimaciones como insumo")
    ultimos = serie.iloc[-rezago:]
    return {
        "metodo": procedimiento["metodo"],
        "formula": procedimiento["formula"],
        "rezago_estacional": rezago,
        "factor_reparto": procedimiento["factor_reparto"],
        "ultimo_mes_observado": f"{serie.index.max():%Y-%m-%d}",
        "valores_base": {f"{f:%Y-%m-%d}": float(v) for f, v in ultimos.items()},
    }


def generar_estimaciones(serie, parametros, cfg, run_id, fecha_generacion):
    procedimiento = cfg["procedimiento"]
    inicio = serie.index.max() + pd.offsets.MonthBegin(1)
    futuros = pd.date_range(inicio, periods=procedimiento["meses_pronostico"], freq="MS")
    filas = []
    for fecha in futuros:
        filas.append({
            "destino": cfg["destino"],
            "periodo_estimado": f"{fecha:%Y-%m-%d}",
            "valor": round(estimar_mes(serie, fecha, parametros["rezago_estacional"], parametros["factor_reparto"]), 2),
            "unidad": procedimiento["unidad"],
            "tipo": "estimado",
            "run_id": run_id,
            "fecha_generacion": fecha_generacion,
            "version_procedimiento": f"{procedimiento['metodo']}-{procedimiento['version']}",
        })
    return pd.DataFrame(filas)


COLUMNAS_SALIDA = ["destino", "periodo_estimado", "valor", "unidad", "tipo", "run_id", "fecha_generacion", "version_procedimiento"]


def integridad_salida(estimaciones, cfg):
    problemas = []
    if list(estimaciones.columns) != COLUMNAS_SALIDA:
        problemas.append("columnas de salida distintas a las esperadas")
    if len(estimaciones) != cfg["procedimiento"]["meses_pronostico"]:
        problemas.append("cantidad de meses estimados distinta a la configurada")
    valores = estimaciones["valor"].to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        problemas.append("estimaciones no finitas")
    if (valores < 0).any():
        problemas.append("estimaciones negativas")
    return problemas


def decidir_aceptacion(controles_ok, problemas_salida, evaluacion, cfg):
    reglas = cfg["aceptacion"]
    criterios = {
        "controles_calidad": controles_ok,
        "integridad_salida": not problemas_salida,
        "mape_dentro_de_umbral": evaluacion["mape_procedimiento"] <= reglas["mape_maximo"],
        "no_peor_que_referencia": evaluacion["razon_mae_referencia"] <= reglas["razon_mae_referencia_maxima"],
    }
    return all(criterios.values()), criterios
