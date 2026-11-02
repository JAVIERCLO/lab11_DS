
import pandas as pd
import pytest

from src import controles, modelo, publicacion
from src.pipeline import RAIZ, cargar_config, ejecutar

CONFIG = RAIZ / "config" / "config.json"


@pytest.fixture
def cfg():
    return cargar_config(CONFIG)


@pytest.fixture
def datos(cfg):
    return pd.read_csv(RAIZ / cfg["fuente"]["ruta"])


@pytest.fixture(autouse=True)
def mlflow_temporal(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")


def estado(resultados, nombre):
    return next(r["estado"] for r in resultados if r["control"] == nombre)


def test_fuente_original_pasa_los_seis_controles(datos, cfg):
    resultados = controles.validar(datos, cfg)
    assert len(resultados) == 6
    assert not controles.hay_fallas(resultados)


def test_columna_ausente_detiene(datos, cfg):
    resultados = controles.validar(datos.drop(columns=["Viajero"]), cfg)
    assert estado(resultados, "columnas_y_tipos") == controles.FALLA


def test_tipo_incompatible_detiene(datos, cfg):
    datos = datos.astype({"Viajero": object})
    datos.loc[5, "Viajero"] = "sin dato"
    assert estado(controles.validar(datos, cfg), "columnas_y_tipos") == controles.FALLA


def test_fecha_invalida_y_mes_sin_cobertura_detienen(datos, cfg):
    invalida = datos.copy()
    invalida.loc[3, "Fecha"] = "2009-13-01"
    assert estado(controles.validar(invalida, cfg), "fechas_y_cobertura") == controles.FALLA
    sin_mes = datos.drop(index=50)
    assert estado(controles.validar(sin_mes, cfg), "fechas_y_cobertura") == controles.FALLA


def test_mes_sin_cobertura_permitido_solo_advierte_y_no_se_vuelve_cero(datos, cfg):
    cfg["periodos"]["meses_faltantes_permitidos"] = ["2010-03-01"]
    datos.loc[datos["Fecha"] == "2010-03-01", "Viajero"] = None
    resultados = controles.validar(datos, cfg)
    assert estado(resultados, "faltantes") == controles.ADVERTENCIA
    serie = controles.preparar_serie(datos, cfg)
    assert pd.Timestamp("2010-03-01") not in serie.index


def test_duplicado_detiene(datos, cfg):
    duplicado = pd.concat([datos, datos.iloc[[10]]])
    assert estado(controles.validar(duplicado, cfg), "unicidad") == controles.FALLA


def test_faltante_no_permitido_detiene(datos, cfg):
    datos.loc[100, "Viajero"] = None
    assert estado(controles.validar(datos, cfg), "faltantes") == controles.FALLA


def test_negativo_infinito_y_factor_cero_detienen(datos, cfg):
    negativo = datos.copy()
    negativo.loc[20, "Viajero"] = -1
    assert estado(controles.validar(negativo, cfg), "valores_y_denominadores") == controles.FALLA
    infinito = datos.copy()
    infinito.loc[20, "Viajero"] = float("inf")
    assert estado(controles.validar(infinito, cfg), "valores_y_denominadores") == controles.FALLA
    cfg["procedimiento"]["factor_reparto"] = 0
    assert estado(controles.validar(datos, cfg), "valores_y_denominadores") == controles.FALLA


def test_serie_diaria_viola_regla_metodologica(datos, cfg):
    diaria = datos.copy()
    diaria.loc[0, "Fecha"] = "2009-01-15"
    assert estado(controles.validar(diaria, cfg), "regla_metodologica") == controles.FALLA
    cfg["fuente"]["tipo"] = "diaria"
    assert estado(controles.validar(datos, cfg), "regla_metodologica") == controles.FALLA


def test_validacion_temporal_no_usa_informacion_futura(datos, cfg):
    serie = controles.preparar_serie(datos, cfg)
    original = modelo.validacion_temporal(serie, cfg)
    alterada = serie.copy()
    alterada.loc["2025-06-01":] = alterada.loc["2025-06-01":] * 10
    cambiada = modelo.validacion_temporal(alterada, cfg)
    antes_de_junio = original["periodo"] < "2025-06-01"
    assert (original["ultimo_mes_usado"] < original["periodo"]).all()
    assert original.loc[antes_de_junio, "estimado"].equals(cambiada.loc[antes_de_junio, "estimado"])


def test_enero_y_febrero_2026_no_se_evaluan(datos, cfg):
    cfg["periodos"]["evaluacion_fin"] = "2026-06-01"
    serie = controles.preparar_serie(datos, cfg)
    periodos = modelo.validacion_temporal(serie, cfg)["periodo"]
    assert not periodos.isin(pd.to_datetime(["2026-01-01", "2026-02-01"])).any()


def test_dos_ejecuciones_son_equivalentes(tmp_path):
    primera = ejecutar(CONFIG, salidas=tmp_path / "a")
    segunda = ejecutar(CONFIG, salidas=tmp_path / "b")
    assert primera["estado"] == segunda["estado"] == "PUBLICADA"
    assert primera["huella_estimaciones"] == segunda["huella_estimaciones"]
    assert primera["evaluacion"] == segunda["evaluacion"]


def test_fallo_conserva_version_aprobada_y_restauracion(tmp_path, datos):
    salidas = tmp_path / "salidas"
    aprobada = ejecutar(CONFIG, salidas=salidas)
    con_duplicado = tmp_path / "duplicado.csv"
    pd.concat([datos, datos.iloc[[-1]]]).to_csv(con_duplicado, index=False)

    fallida = ejecutar(CONFIG, fuente=con_duplicado, salidas=salidas)
    assert fallida["estado"] == "FALLIDA"
    assert "evaluacion" not in fallida
    assert publicacion.leer_vigente(salidas)["run_id"] == aprobada["run_id"]

    corregida = ejecutar(CONFIG, salidas=salidas)
    assert publicacion.leer_vigente(salidas)["run_id"] == corregida["run_id"]
    publicacion.restaurar(salidas, aprobada["run_id"])
    estimaciones, vigente = publicacion.leer_estimaciones_vigentes(salidas)
    assert vigente["run_id"] == aprobada["run_id"]
    assert estimaciones is not None


def test_version_alterada_no_se_restaura(tmp_path):
    salidas = tmp_path / "salidas"
    aprobada = ejecutar(CONFIG, salidas=salidas)
    archivo = salidas / "publicado" / "versiones" / aprobada["run_id"] / "estimaciones.csv"
    archivo.write_text(archivo.read_text().replace("2026-07-01", "2026-07-02"))
    with pytest.raises(ValueError):
        publicacion.restaurar(salidas, aprobada["run_id"])
