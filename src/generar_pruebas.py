import json
from pathlib import Path

import pandas as pd

from src.pipeline import RAIZ, cargar_config


def generar(destino=RAIZ / "data" / "pruebas"):
    cfg = cargar_config(RAIZ / "config" / "config.json")
    original = pd.read_csv(RAIZ / cfg["fuente"]["ruta"])
    destino.mkdir(parents=True, exist_ok=True)

    original.drop(columns=["Viajero"]).to_csv(destino / "sin_columna.csv", index=False)
    pd.concat([original, original.iloc[[-1]]]).to_csv(destino / "duplicado.csv", index=False)
    original.drop(index=original.index[120]).to_csv(destino / "mes_faltante.csv", index=False)
    negativos = original.copy()
    negativos.loc[negativos.index[-3], "Viajero"] = -5
    negativos.to_csv(destino / "negativo.csv", index=False)

    cfg_factor = json.loads(json.dumps(cfg))
    cfg_factor["procedimiento"]["factor_reparto"] = 0
    (destino / "config_factor_cero.json").write_text(json.dumps(cfg_factor, indent=2, ensure_ascii=False))
    return sorted(p.name for p in destino.iterdir())


if __name__ == "__main__":
    for nombre in generar():
        print(nombre)
