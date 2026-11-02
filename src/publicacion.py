import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def carpeta_publicado(salidas):
    return Path(salidas) / "publicado"


def huella(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def escribir_json_atomico(ruta, contenido):
    temporal = Path(str(ruta) + ".tmp")
    temporal.write_text(json.dumps(contenido, indent=2, ensure_ascii=False))
    # os.replace es atómico, el tablero nunca lee un puntero a medio escribir
    os.replace(temporal, ruta)


def leer_vigente(salidas):
    ruta = carpeta_publicado(salidas) / "vigente.json"
    if not ruta.exists():
        return None
    return json.loads(ruta.read_text())


def leer_estimaciones_vigentes(salidas):
    vigente = leer_vigente(salidas)
    if vigente is None:
        return None, None
    archivo = carpeta_publicado(salidas) / vigente["archivo"]
    if not archivo.exists() or huella(archivo) != vigente["sha256"]:
        return None, vigente
    return pd.read_csv(archivo), vigente


def listar_versiones(salidas):
    carpeta = carpeta_publicado(salidas) / "versiones"
    if not carpeta.exists():
        return []
    return sorted(p.name for p in carpeta.iterdir() if (p / "metadatos.json").exists())


def registrar_historial(salidas, evento):
    with open(carpeta_publicado(salidas) / "historial.jsonl", "a") as archivo:
        archivo.write(json.dumps(evento, ensure_ascii=False) + "\n")


def publicar(salidas, run_id, estimaciones, metadatos):
    version = carpeta_publicado(salidas) / "versiones" / run_id
    version.mkdir(parents=True, exist_ok=True)
    ruta_csv = version / "estimaciones.csv"
    estimaciones.to_csv(ruta_csv, index=False)
    metadatos = {**metadatos, "sha256": huella(ruta_csv)}
    (version / "metadatos.json").write_text(json.dumps(metadatos, indent=2, ensure_ascii=False))
    anterior = leer_vigente(salidas)
    vigente = {
        "run_id": run_id,
        "archivo": f"versiones/{run_id}/estimaciones.csv",
        "sha256": metadatos["sha256"],
        "publicado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_anterior": anterior["run_id"] if anterior else None,
        "metadatos": metadatos,
    }
    escribir_json_atomico(carpeta_publicado(salidas) / "vigente.json", vigente)
    registrar_historial(salidas, {"accion": "publicacion", "run_id": run_id, "fecha": vigente["publicado_en"]})
    return vigente


def restaurar(salidas, run_id):
    version = carpeta_publicado(salidas) / "versiones" / run_id
    if not (version / "metadatos.json").exists():
        raise FileNotFoundError(f"No existe la versión aprobada {run_id}. Disponibles: {listar_versiones(salidas)}")
    metadatos = json.loads((version / "metadatos.json").read_text())
    if huella(version / "estimaciones.csv") != metadatos["sha256"]:
        raise ValueError(f"El archivo de la versión {run_id} fue modificado, no se restaura")
    anterior = leer_vigente(salidas)
    vigente = {
        "run_id": run_id,
        "archivo": f"versiones/{run_id}/estimaciones.csv",
        "sha256": metadatos["sha256"],
        "publicado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version_anterior": anterior["run_id"] if anterior else None,
        "restaurada": True,
        "metadatos": metadatos,
    }
    escribir_json_atomico(carpeta_publicado(salidas) / "vigente.json", vigente)
    registrar_historial(salidas, {"accion": "restauracion", "run_id": run_id, "reemplaza_a": vigente["version_anterior"], "fecha": vigente["publicado_en"]})
    return vigente
