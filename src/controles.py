import numpy as np
import pandas as pd

PASA = "PASA"
ADVERTENCIA = "ADVERTENCIA"
FALLA = "FALLA"


def resultado(control, estado, detalle):
    return {"control": control, "estado": estado, "detalle": detalle}


def meses_requeridos(cfg):
    periodos = cfg["periodos"]
    return pd.date_range(periodos["cobertura_inicio"], periodos["cobertura_fin"], freq="MS")


def control_columnas_y_tipos(df, cfg):
    fuente = cfg["fuente"]
    requeridas = [fuente["columna_fecha"], fuente["columna_valor"]]
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        return resultado("columnas_y_tipos", FALLA, f"Faltan columnas indispensables: {faltantes}")
    valores = df[fuente["columna_valor"]]
    no_numericos = valores.notna() & pd.to_numeric(valores, errors="coerce").isna()
    if no_numericos.any():
        return resultado("columnas_y_tipos", FALLA, f"{int(no_numericos.sum())} valores no numéricos en {fuente['columna_valor']}")
    extra = [c for c in df.columns if c not in requeridas]
    if extra:
        return resultado("columnas_y_tipos", ADVERTENCIA, f"Columnas adicionales ignoradas: {extra}")
    return resultado("columnas_y_tipos", PASA, "Columnas Fecha y Viajero presentes con tipos compatibles")


def control_fechas_y_cobertura(df, cfg):
    fechas = pd.to_datetime(df[cfg["fuente"]["columna_fecha"]], errors="coerce")
    invalidas = int(fechas.isna().sum())
    if invalidas:
        return resultado("fechas_y_cobertura", FALLA, f"{invalidas} fechas inválidas")
    requeridos = meses_requeridos(cfg)
    permitidos = set(pd.to_datetime(cfg["periodos"]["meses_faltantes_permitidos"]))
    ausentes = requeridos.difference(fechas)
    criticos = [m for m in ausentes if m not in permitidos]
    if criticos:
        meses = [m.strftime("%Y-%m") for m in criticos]
        return resultado("fechas_y_cobertura", FALLA, f"Meses sin información: {meses}")
    fuera = fechas[(fechas < requeridos.min()) | (fechas > requeridos.max())]
    if len(ausentes) or len(fuera):
        return resultado("fechas_y_cobertura", ADVERTENCIA, f"{len(ausentes)} meses ausentes permitidos y {len(fuera)} fechas fuera de la cobertura configurada")
    return resultado("fechas_y_cobertura", PASA, f"{len(requeridos)} meses entre {requeridos.min():%Y-%m} y {requeridos.max():%Y-%m}")


def control_unicidad(df, cfg):
    fechas = pd.to_datetime(df[cfg["fuente"]["columna_fecha"]], errors="coerce")
    periodos = fechas.dt.to_period("M")
    duplicados = periodos[periodos.duplicated(keep=False)].dropna().unique()
    if len(duplicados):
        return resultado("unicidad", FALLA, f"Meses repetidos según la clave destino-mes: {[str(p) for p in duplicados]}")
    return resultado("unicidad", PASA, "Una fila por mes")


def control_faltantes(df, cfg):
    fuente = cfg["fuente"]
    fechas = pd.to_datetime(df[fuente["columna_fecha"]], errors="coerce")
    valores = pd.to_numeric(df[fuente["columna_valor"]], errors="coerce")
    permitidos = set(pd.to_datetime(cfg["periodos"]["meses_faltantes_permitidos"]))
    meses_vacios = fechas[valores.isna()]
    criticos = [m for m in meses_vacios if m not in permitidos]
    if criticos:
        return resultado("faltantes", FALLA, f"Valores ausentes que impiden estimar: {[f'{m:%Y-%m}' for m in criticos]}")
    if len(meses_vacios):
        return resultado("faltantes", ADVERTENCIA, f"{len(meses_vacios)} ausencias permitidas, no se imputan como cero")
    return resultado("faltantes", PASA, "Sin valores ausentes")


def control_valores_y_denominadores(df, cfg):
    fuente = cfg["fuente"]
    fechas = pd.to_datetime(df[fuente["columna_fecha"]], errors="coerce")
    valores = pd.to_numeric(df[fuente["columna_valor"]], errors="coerce")
    presentes = valores.dropna()
    problemas = []
    if (~np.isfinite(presentes)).any():
        problemas.append(f"{int((~np.isfinite(presentes)).sum())} valores no finitos")
    if (presentes < 0).any():
        problemas.append(f"{int((presentes < 0).sum())} conteos negativos")
    factor = cfg["procedimiento"]["factor_reparto"]
    if not np.isfinite(factor) or factor <= 0:
        problemas.append(f"factor de reparto inválido ({factor})")
    periodos = cfg["periodos"]
    en_evaluacion = (fechas >= periodos["evaluacion_inicio"]) & (fechas <= periodos["evaluacion_fin"])
    if (valores[en_evaluacion] == 0).any():
        problemas.append("observados iguales a cero en evaluación, el MAPE quedaría con denominador cero")
    if problemas:
        return resultado("valores_y_denominadores", FALLA, "; ".join(problemas))
    if not np.allclose(presentes, presentes.round()):
        return resultado("valores_y_denominadores", ADVERTENCIA, "Valores con decimales, la fuente parece ajustada y no un conteo directo")
    return resultado("valores_y_denominadores", PASA, "Conteos finitos y no negativos, factor de reparto positivo")


def control_regla_metodologica(df, cfg):
    fuente = cfg["fuente"]
    if fuente["tipo"] != "mensual":
        return resultado("regla_metodologica", FALLA, f"La fuente declarada es '{fuente['tipo']}', se exige la fuente mensual")
    prohibidas = [c for c in fuente["columnas_prohibidas"] if c in df.columns]
    if prohibidas:
        return resultado("regla_metodologica", FALLA, f"La fuente trae columnas de presencias diarias: {prohibidas}")
    fechas = pd.to_datetime(df[fuente["columna_fecha"]], errors="coerce").dropna()
    no_inicio_mes = fechas[fechas.dt.day != 1]
    if len(no_inicio_mes):
        return resultado("regla_metodologica", FALLA, f"{len(no_inicio_mes)} fechas no son inicio de mes, la serie parece construida con registros diarios")
    return resultado("regla_metodologica", PASA, "Serie tomada de la fuente mensual, sin sumar presencias diarias")


CONTROLES = [
    control_columnas_y_tipos,
    control_fechas_y_cobertura,
    control_unicidad,
    control_faltantes,
    control_valores_y_denominadores,
    control_regla_metodologica,
]


def validar(df, cfg):
    resultados = []
    for control in CONTROLES:
        res = control(df, cfg)
        resultados.append(res)
        # Sin columnas indispensables los demás controles no pueden evaluarse
        if control is control_columnas_y_tipos and res["estado"] == FALLA:
            break
    return resultados


def hay_fallas(resultados):
    return any(r["estado"] == FALLA for r in resultados)


def preparar_serie(df, cfg):
    fuente = cfg["fuente"]
    serie = pd.Series(
        pd.to_numeric(df[fuente["columna_valor"]], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[fuente["columna_fecha"]]),
        name="observado",
    ).sort_index()
    requeridos = meses_requeridos(cfg)
    return serie.loc[requeridos.min():requeridos.max()].dropna()


def resumen_calidad(df, cfg):
    fuente = cfg["fuente"]
    if fuente["columna_fecha"] not in df.columns or fuente["columna_valor"] not in df.columns:
        return {"filas": len(df)}
    fechas = pd.to_datetime(df[fuente["columna_fecha"]], errors="coerce")
    valores = pd.to_numeric(df[fuente["columna_valor"]], errors="coerce")
    return {
        "filas": len(df),
        "faltantes": int(valores.isna().sum()),
        "duplicados": int(fechas.dt.to_period("M").duplicated().sum()),
        "meses_cubiertos": int(meses_requeridos(cfg).isin(fechas).sum()),
    }
