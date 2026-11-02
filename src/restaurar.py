import argparse
import json
import sys
from pathlib import Path

from src import publicacion
from src.pipeline import RAIZ, cargar_config


def main():
    parser = argparse.ArgumentParser(description="Restaura una versión aprobada como versión vigente del tablero")
    parser.add_argument("run_id", nargs="?", help="Versión a restaurar. Sin argumento solo lista las disponibles")
    parser.add_argument("--config", default=str(RAIZ / "config" / "config.json"))
    args = parser.parse_args()
    salidas = RAIZ / cargar_config(args.config)["rutas"]["salidas"]
    vigente = publicacion.leer_vigente(salidas)
    if not args.run_id:
        print("Versión vigente:", vigente["run_id"] if vigente else "ninguna")
        for version in publicacion.listar_versiones(salidas):
            print(" ", version)
        return 0
    try:
        nueva = publicacion.restaurar(salidas, args.run_id)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1
    print(json.dumps({k: nueva[k] for k in ["run_id", "version_anterior", "publicado_en"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
