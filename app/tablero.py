import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import publicacion

COLOR_OBSERVADO = "#1f4e79"
COLOR_ESTIMADO = "#d9822b"

st.set_page_config(page_title="Observatorio de Turismo Sostenible", layout="wide")
cfg = json.loads((RAIZ / "config" / "config.json").read_text())
salidas = RAIZ / cfg["rutas"]["salidas"]


# El tablero solo lee archivos ya publicados, cambiar un filtro nunca vuelve a ajustar el modelo
@st.cache_data
def cargar_observados(ruta, modificado):
    datos = pd.read_csv(ruta, parse_dates=[cfg["fuente"]["columna_fecha"]])
    return datos.rename(columns={cfg["fuente"]["columna_fecha"]: "periodo", cfg["fuente"]["columna_valor"]: "valor"})


@st.cache_data
def cargar_vigente(sha):
    return publicacion.leer_estimaciones_vigentes(salidas)


st.title("Viajeros mensuales")
st.caption(f"Observatorio de Turismo Sostenible · {cfg['destino']}")

vigente = publicacion.leer_vigente(salidas)
estimaciones, vigente = cargar_vigente(vigente["sha256"] if vigente else None)
ruta_fuente = RAIZ / cfg["fuente"]["ruta"]
observados = cargar_observados(ruta_fuente, ruta_fuente.stat().st_mtime) if ruta_fuente.exists() else None

if estimaciones is None:
    st.error("Las estimaciones no están disponibles. Todavía no existe una versión aprobada o el archivo publicado no pasó la verificación de integridad.")
else:
    estimaciones["periodo"] = pd.to_datetime(estimaciones["periodo_estimado"])
    meta = vigente["metadatos"]
    metricas = meta["metricas"]
    actualizado = pd.Timestamp(vigente["publicado_en"]).strftime("%d/%m/%Y")
    inicio_eval, fin_eval = meta["periodo_evaluado"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Última actualización", actualizado)
    col2.metric("Error promedio en validación (MAPE)", f"{metricas['mape_procedimiento']:.1f} %")
    col3.metric("Próximo mes estimado", f"{estimaciones['valor'].iloc[0]:,.0f}", help=f"Mes {estimaciones['periodo'].iloc[0]:%Y-%m}")
    st.markdown(
        f"**Cómo leer el error.** Se ocultaron los meses de {inicio_eval} a {fin_eval} y se estimaron con la información "
        f"disponible antes de cada mes. En promedio la estimación se alejó **{metricas['mape_procedimiento']:.1f} %** "
        f"({metricas['mae_procedimiento']:,.0f} viajeros) del valor observado. Repetir el valor del mes anterior "
        f"se habría alejado {metricas['mape_referencia']:.1f} %."
    )

    if observados is not None:
        anio_min, anio_max = int(observados["periodo"].dt.year.min()), int(estimaciones["periodo"].dt.year.max())
        desde, hasta = st.slider("Años a mostrar", anio_min, anio_max, (max(anio_min, anio_max - 5), anio_max))
        visibles = observados[observados["periodo"].dt.year.between(desde, hasta)]
        figura = go.Figure()
        figura.add_trace(go.Scatter(x=visibles["periodo"], y=visibles["valor"], name="Observado", mode="lines", line=dict(color=COLOR_OBSERVADO, width=2)))
        ultimo = observados.iloc[[-1]]
        puente = pd.concat([ultimo[["periodo", "valor"]], estimaciones[["periodo", "valor"]]])
        figura.add_trace(go.Scatter(x=puente["periodo"], y=puente["valor"], name="Estimado", mode="lines+markers", line=dict(color=COLOR_ESTIMADO, width=2, dash="dash")))
        figura.add_vrect(x0=estimaciones["periodo"].min() - pd.Timedelta(days=15), x1=estimaciones["periodo"].max() + pd.Timedelta(days=15), fillcolor=COLOR_ESTIMADO, opacity=0.08, line_width=0, annotation_text="Estimación", annotation_position="top left")
        figura.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Viajeros por mes", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(figura, use_container_width=True)

    tabla = estimaciones.assign(
        Mes=estimaciones["periodo"].dt.strftime("%Y-%m"),
        Viajeros=estimaciones["valor"].map("{:,.0f}".format),
        Tipo="Estimado",
    )[["Mes", "Viajeros", "Tipo"]]
    st.subheader("Estimaciones")
    st.dataframe(tabla, hide_index=True, use_container_width=True)

    st.warning(
        "**Limitaciones.** Las estimaciones repiten el comportamiento del mismo mes del año anterior, por lo que no anticipan "
        "cambios bruscos como cierres de fronteras, crisis o eventos extraordinarios. Los valores de la fuente tienen decimales "
        "porque son una serie ajustada y no un conteo directo de registros. Enero y febrero de 2026 no se usaron para medir el error "
        "porque no hay observaciones comparables para esos meses. Un error de validación bajo no garantiza el mismo desempeño en meses futuros."
    )
    restaurada = " (versión restaurada)" if vigente.get("restaurada") else ""
    st.caption(f"Versión publicada {vigente['run_id']}{restaurada} · procedimiento {meta['version_procedimiento']} · unidad: {meta['unidad']}")
