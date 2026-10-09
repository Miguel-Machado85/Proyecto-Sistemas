"""Exporta un catálogo legible de la metadata almacenada en Chroma.

No genera embeddings ni modifica la colección: solo agrupa los fragmentos
indexados por archivo para obtener una ficha por documento.
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from langchain_chroma import Chroma

from agente_sigem.config import config


CAMPOS_PDF = (
    "categoria",
    "fecha_expedicion",
    "total_pages",
    "requirio_ocr",
    "author",
    "creator",
    "producer",
    "creationdate",
    "moddate",
    "keywords",
    "title",
    "subject",
)


def primer_valor(metadatas: list[dict], campo: str):
    """Devuelve el primer valor no vacío de un campo compartido por los fragmentos."""
    return next((metadata[campo] for metadata in metadatas if metadata.get(campo) not in (None, "")), None)


def crear_catalogo() -> dict:
    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )
    resultado = vector_store.get(include=["metadatas"])
    grupos = defaultdict(list)
    for metadata in resultado["metadatas"]:
        metadata = metadata or {}
        clave = metadata.get("nombre_archivo") or metadata.get("source") or "sin_archivo"
        grupos[clave].append(metadata)

    documentos = []
    for archivo, metadatas in grupos.items():
        paginas = sorted(
            {
                str(metadata.get("page_label") or int(metadata["page"]) + 1)
                for metadata in metadatas
                if metadata.get("page_label") is not None or metadata.get("page") is not None
            },
            key=lambda pagina: int(pagina) if pagina.isdigit() else pagina,
        )
        ficha = {
            "archivo": archivo,
            "titulo": primer_valor(metadatas, "titulo"),
            "descripcion": primer_valor(metadatas, "descripcion"),
            "fragmentos_indexados": len(metadatas),
            "paginas_indexadas": paginas,
            "metadata": {campo: primer_valor(metadatas, campo) for campo in CAMPOS_PDF},
        }
        documentos.append(ficha)

    documentos.sort(key=lambda documento: ((documento["titulo"] or "").lower(), documento["archivo"].lower()))
    return {
        "coleccion": config.COLLECTION_NAME,
        "generado_en_utc": datetime.now(timezone.utc).isoformat(),
        "fragmentos_totales": len(resultado["metadatas"]),
        "documentos_unicos": len(documentos),
        "documentos": documentos,
    }


def escribir_csv(documentos: list[dict], destino: Path) -> None:
    campos = ["archivo", "titulo", "descripcion", "fragmentos_indexados", "paginas_indexadas", *CAMPOS_PDF]
    with destino.open("w", newline="", encoding="utf-8-sig") as archivo_csv:
        escritor = csv.DictWriter(archivo_csv, fieldnames=campos)
        escritor.writeheader()
        for documento in documentos:
            fila = {
                "archivo": documento["archivo"],
                "titulo": documento["titulo"],
                "descripcion": documento["descripcion"],
                "fragmentos_indexados": documento["fragmentos_indexados"],
                "paginas_indexadas": ", ".join(documento["paginas_indexadas"]),
                **documento["metadata"],
            }
            escritor.writerow(fila)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=".", help="Carpeta donde se escriben el JSON y CSV.")
    args = parser.parse_args()

    destino = Path(args.output_dir)
    destino.mkdir(parents=True, exist_ok=True)
    catalogo = crear_catalogo()

    (destino / "catalogo_chroma.json").write_text(
        json.dumps(catalogo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    escribir_csv(catalogo["documentos"], destino / "catalogo_chroma.csv")
    print(
        f"Exportados {catalogo['documentos_unicos']} documentos y "
        f"{catalogo['fragmentos_totales']} fragmentos en {destino.resolve()}"
    )


if __name__ == "__main__":
    main()
