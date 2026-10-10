"""Evaluación de recuperación (sin LLM): hit@8 y MRR por categoría usando recuperar_sigem.

Solo usa embeddings (GOOGLE_API_KEY) y el Chroma local (chroma_db/). No llama a ningún chat LLM.
Sirve para medir la versión actual (línea base) y, más adelante, la versión con reranker.

Uso (desde backend/):
    uv run python evaluacion/evaluar_recuperacion.py
    uv run python evaluacion/evaluar_recuperacion.py --sin-reranker --salida evaluacion/recuperacion_sin_reranker.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RAIZ_BACKEND = Path(__file__).resolve().parent.parent
K = 8


def cargar_casos(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def consulta_del_caso(caso: dict) -> str:
    """Pregunta autónoma equivalente; si no se definió, el último mensaje del caso."""
    return caso.get("consulta_recuperacion") or caso["turnos"][-1]["mensaje"]


def archivos_en_orden(documentos) -> list[str]:
    """Archivos únicos en el orden en que aparecen los fragmentos recuperados."""
    vistos = []
    for documento in documentos:
        metadata = documento.metadata or {}
        archivo = str(metadata.get("nombre_archivo") or metadata.get("source") or "")
        if archivo and archivo not in vistos:
            vistos.append(archivo)
    return vistos


def evaluar_caso(recuperar, caso: dict) -> dict:
    consulta = consulta_del_caso(caso)
    inicio = time.perf_counter()
    documentos = recuperar(consulta)[:K]
    tiempo_ms = (time.perf_counter() - inicio) * 1000
    archivos = archivos_en_orden(documentos)
    esperados = set(caso["archivos_esperados"])
    posiciones = [i for i, archivo in enumerate(archivos, start=1) if archivo in esperados]
    primera = posiciones[0] if posiciones else None
    return {
        "id": caso["id"],
        "categoria": caso["categoria"],
        "perfil": caso["perfil"],
        "consulta": consulta,
        "esperados": sorted(esperados),
        "recuperados": archivos,
        "hit": primera is not None,
        "rr": (1.0 / primera) if primera else 0.0,
        "tiempo_ms": round(tiempo_ms),
    }


def resumir(resultados: list[dict]) -> dict:
    por_categoria = defaultdict(list)
    for resultado in resultados:
        por_categoria[resultado["categoria"]].append(resultado)

    def metricas(grupo: list[dict]) -> dict:
        return {
            "casos": len(grupo),
            f"hit@{K}": round(sum(r["hit"] for r in grupo) / len(grupo), 3),
            "mrr": round(sum(r["rr"] for r in grupo) / len(grupo), 3),
            "tiempo_ms_promedio": round(sum(r["tiempo_ms"] for r in grupo) / len(grupo)),
        }

    resumen = {categoria: metricas(grupo) for categoria, grupo in sorted(por_categoria.items())}
    resumen["GLOBAL"] = metricas(resultados)
    return resumen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(RAIZ_BACKEND / "evaluacion" / "dataset.jsonl"))
    parser.add_argument("--salida", default=str(RAIZ_BACKEND / "evaluacion" / "linea_base_recuperacion.json"))
    parser.add_argument("--sin-reranker", action="store_true",
                        help="Apaga el reranker (RERANKER_HABILITADO=false). Sin efecto en el código anterior a la migración.")
    parser.add_argument("--con-referencia", action="store_true",
                        help="Usa referencias.recuperar_con_referencia (la referencia exacta gana, como en el código anterior). "
                             "Equivale al comportamiento de buscar_en_sigem y es comparable con la línea base.")
    args = parser.parse_args()

    # config.py usa rutas relativas (./chroma_db): se ejecuta siempre desde backend/.
    os.chdir(RAIZ_BACKEND)
    sys.path.insert(0, str(RAIZ_BACKEND))
    if args.sin_reranker:
        os.environ["RERANKER_HABILITADO"] = "false"

    from agente_sigem.config import config

    if not config.GOOGLE_API_KEY:
        print("ERROR: falta GOOGLE_API_KEY (necesaria para los embeddings de consulta). No se calculó nada.")
        return 2
    chroma = Path(config.CHROMA_PERSIST_DIR)
    if not (chroma / "chroma.sqlite3").exists():
        print(f"ERROR: no hay un Chroma local en {chroma.resolve()}. No se calculó nada.")
        return 2

    if args.con_referencia:
        from agente_sigem.referencias import recuperar_con_referencia as recuperar_sigem
    else:
        from agente_sigem.retrieval import recuperar_sigem

    if args.sin_reranker is False:
        # Carga el modelo antes de medir para que el tiempo por consulta no incluya la descarga/carga.
        from agente_sigem import reranker
        inicio = time.perf_counter()
        print(f"Reranker precargado: {reranker.precargar()} ({time.perf_counter() - inicio:.1f}s)")

    casos = [c for c in cargar_casos(Path(args.dataset)) if c.get("archivos_esperados")]
    print(f"Evaluando {len(casos)} casos con archivos_esperados (k={K}, reranker={'no' if args.sin_reranker else 'según config'}, con_referencia={args.con_referencia})")

    resultados = []
    for caso in casos:
        resultado = evaluar_caso(recuperar_sigem, caso)
        resultados.append(resultado)
        marca = "OK " if resultado["hit"] else "MISS"
        print(f"  [{marca}] {caso['id']:<14} rr={resultado['rr']:.2f}  {resultado['consulta'][:70]}")

    resumen = resumir(resultados)
    print("\nResumen por categoría")
    for categoria, m in resumen.items():
        print(f"  {categoria:<34} n={m['casos']:<3} hit@{K}={m[f'hit@{K}']:.2f}  mrr={m['mrr']:.2f}  {m['tiempo_ms_promedio']} ms")

    salida = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "k": K,
        "sin_reranker": args.sin_reranker,
        "con_referencia": args.con_referencia,
        "embedding_model": config.EMBEDDING_MODEL,
        "resumen": resumen,
        "casos": resultados,
    }
    Path(args.salida).write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
