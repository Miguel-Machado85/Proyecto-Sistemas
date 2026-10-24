"""Migra la colección de `backend/chroma_db` (embebido) a un servidor Chroma, SIN regenerar embeddings.

- El origen solo se LEE: se trabaja sobre una copia temporal de la carpeta, porque abrir un
  PersistentClient puede escribir en disco (migraciones internas de sqlite).
- Lee en páginas con ids, documentos, metadatos y embeddings; escribe con `upsert` (idempotente:
  se puede volver a ejecutar si se corta).
- La colección destino se crea con la misma configuración de distancia que la origen; si no coincide,
  la búsqueda semántica cambiaría en silencio, así que se aborta.
- La verificación compara conteo, ids y el top-8 de fragmentos al azar usando su propio embedding
  guardado como consulta (el destino no puede recuperar peor que el origen frente al top-8 exacto):
  no llama a Gemini ni gasta cuota.

Uso (desde backend/, con el servidor de CHROMA_HOST/CHROMA_PORT arriba):
    uv run python scripts/migrar_chroma.py                 # migra y verifica
    uv run python scripts/migrar_chroma.py --solo-verificar
"""

import argparse
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterator

import chromadb
import numpy as np

RAIZ_BACKEND = Path(__file__).resolve().parent.parent
if str(RAIZ_BACKEND) not in sys.path:
    sys.path.insert(0, str(RAIZ_BACKEND))

TAMANO_PAGINA = 500
MUESTRAS_VERIFICACION = 10
TOP_K = 8
INCLUIR = ["embeddings", "documents", "metadatas"]


def abrir_origen_solo_lectura(ruta: Path) -> tuple[chromadb.ClientAPI, Path]:
    """Abre una COPIA de la carpeta local; devuelve el cliente y la carpeta temporal a borrar."""
    if not (ruta / "chroma.sqlite3").exists():
        raise FileNotFoundError(f"No hay un índice Chroma en {ruta}.")
    temporal = Path(tempfile.mkdtemp(prefix="chroma_origen_"))
    copia = temporal / "chroma_db"
    shutil.copytree(ruta, copia)
    return chromadb.PersistentClient(path=str(copia)), temporal


def leer_paginas(coleccion, tamano: int = TAMANO_PAGINA) -> Iterator[dict]:
    """Páginas de la colección con ids, documentos, metadatos y embeddings (orden estable por offset)."""
    total = coleccion.count()
    for desplazamiento in range(0, total, tamano):
        pagina = coleccion.get(limit=tamano, offset=desplazamiento, include=INCLUIR)
        if pagina["ids"]:
            yield pagina


def configuracion_hnsw(coleccion) -> dict:
    """Parámetros HNSW explícitos de la colección (incluida la métrica de distancia)."""
    hnsw = dict((coleccion.configuration or {}).get("hnsw") or {})
    return {clave: valor for clave, valor in hnsw.items() if valor is not None}


def crear_destino(cliente_destino, origen, nombre: str | None = None):
    """Crea (o reutiliza) la colección destino con la misma distancia que la origen."""
    config_origen = configuracion_hnsw(origen)
    destino = cliente_destino.get_or_create_collection(
        name=nombre or origen.name,
        metadata=origen.metadata or None,
        configuration={"hnsw": config_origen} if config_origen else None,
    )
    espacio_origen = config_origen.get("space")
    espacio_destino = configuracion_hnsw(destino).get("space")
    if espacio_origen != espacio_destino:
        raise RuntimeError(
            f"La distancia del destino ({espacio_destino}) no coincide con la del origen ({espacio_origen}). "
            "No se migra: la búsqueda semántica cambiaría en silencio."
        )
    return destino


def migrar(origen, destino, tamano: int = TAMANO_PAGINA) -> int:
    """Copia todo con upsert por páginas. Devuelve cuántos registros procesó."""
    procesados = 0
    total = origen.count()
    for pagina in leer_paginas(origen, tamano):
        destino.upsert(
            ids=pagina["ids"],
            embeddings=pagina["embeddings"],
            documents=pagina["documents"],
            metadatas=pagina["metadatas"],
        )
        procesados += len(pagina["ids"])
        print(f"  {procesados}/{total} registros", flush=True)
    return procesados


def ids_de(coleccion) -> list[str]:
    ids: list[str] = []
    for desplazamiento in range(0, coleccion.count(), TAMANO_PAGINA):
        ids.extend(coleccion.get(limit=TAMANO_PAGINA, offset=desplazamiento, include=[])["ids"])
    return ids


def verificar(origen, destino, muestras: int = MUESTRAS_VERIFICACION, semilla: int = 7) -> list[str]:
    """Devuelve la lista de problemas encontrados (vacía = verificación correcta)."""
    problemas: list[str] = []
    if origen.count() != destino.count():
        problemas.append(f"Conteo distinto: origen {origen.count()}, destino {destino.count()}.")
    ids_origen, ids_destino = set(ids_de(origen)), set(ids_de(destino))
    if ids_origen != ids_destino:
        problemas.append(
            f"Ids distintos: faltan en destino {len(ids_origen - ids_destino)}, sobran {len(ids_destino - ids_origen)}."
        )
    if configuracion_hnsw(origen).get("space") != configuracion_hnsw(destino).get("space"):
        problemas.append("La métrica de distancia es distinta.")

    # El índice HNSW es aproximado y se construye distinto en cada instalación: dos índices con los
    # mismos vectores pueden diferir en algún vecino casi empatado. Por eso el top-8 de cada lado se
    # compara con el top-8 EXACTO (fuerza bruta): el destino no puede ser peor que el origen.
    ids_exactos, matriz = [], []
    for pagina in leer_paginas(origen):
        ids_exactos.extend(pagina["ids"])
        matriz.extend(pagina["embeddings"])
    matriz = np.asarray(matriz, dtype=np.float32)

    comunes = sorted(ids_origen & ids_destino)
    elegidos = random.Random(semilla).sample(comunes, min(muestras, len(comunes)))
    idénticos = 0
    for identificador in elegidos:
        posicion = ids_exactos.index(identificador)
        consulta = matriz[posicion]
        distancias = ((matriz - consulta) ** 2).sum(axis=1)
        exacto = {ids_exactos[i] for i in np.argsort(distancias)[:TOP_K]}
        top_origen = origen.query(query_embeddings=[consulta.tolist()], n_results=TOP_K)["ids"][0]
        top_destino = destino.query(query_embeddings=[consulta.tolist()], n_results=TOP_K)["ids"][0]
        idénticos += top_origen == top_destino
        if len(exacto & set(top_destino)) < len(exacto & set(top_origen)):
            problemas.append(f"El destino recupera peor que el origen para {identificador}: {top_destino} vs {top_origen}.")
    print(f"  top-{TOP_K} idéntico en {idénticos}/{len(elegidos)} muestras; el resto difiere solo por vecinos casi empatados.")
    return problemas


def main() -> int:
    from agente_sigem.config import config

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origen", default=config.CHROMA_PERSIST_DIR, help="Carpeta chroma_db de origen (solo lectura).")
    parser.add_argument("--solo-verificar", action="store_true", help="No copia; solo compara origen y destino.")
    args = parser.parse_args()

    origen_cliente, temporal = abrir_origen_solo_lectura(Path(args.origen).resolve())
    try:
        origen = origen_cliente.get_collection(config.COLLECTION_NAME)
        destino_cliente = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT, ssl=config.CHROMA_SSL)
        destino_cliente.heartbeat()
        print(f"Origen: {args.origen} · {origen.count()} registros · distancia {configuracion_hnsw(origen).get('space')}")
        print(f"Destino: {config.CHROMA_HOST}:{config.CHROMA_PORT}")
        destino = crear_destino(destino_cliente, origen)
        if not args.solo_verificar:
            migrar(origen, destino)
        problemas = verificar(origen, destino)
    finally:
        shutil.rmtree(temporal, ignore_errors=True)

    if problemas:
        print("VERIFICACIÓN FALLIDA:")
        for problema in problemas:
            print(f"  - {problema}")
        return 1
    print(
        f"VERIFICACIÓN CORRECTA: {destino.count()} registros, mismos ids, misma distancia y "
        f"top-{TOP_K} del destino no peor que el del origen en {MUESTRAS_VERIFICACION} muestras."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
