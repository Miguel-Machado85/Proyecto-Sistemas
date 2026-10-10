"""Evaluación de extremo a extremo: corre el grafo completo sobre el dataset (sección 13).

Por cada caso crea un thread nuevo y ejecuta todos sus turnos en orden (CHECKPOINTER=memory), de modo
que los casos de seguimiento y de búsqueda global con documento activo prueban la memoria real.
Mide sobre el ÚLTIMO turno de cada caso:
  - intencion_ok:  la intención del analista coincide con `intencion_esperada`.
  - archivos_ok:   algún archivo de `archivos_esperados` aparece en `fuentes` (o en `candidatos`
                   cuando el caso espera candidatos).
  - candidatos_ok: hay candidatos cuando el caso los espera (`candidatos_esperados`).
  - alerta_ok:     `es_alerta` es True cuando se espera `no_comprensible`.

IMPORTANTE (cuota): cada consulta con búsqueda gasta entre 3 y 6 llamadas a Gemini y el dataset tiene
unos 50 turnos. El plan gratuito de `gemini-3.5-flash` permite ~20 solicitudes/día; para una corrida
completa hace falta facturación, o bajar los modelos pesados con variables de entorno, p. ej.:
    INVESTIGADOR_MODEL=gemini-3.5-flash-lite REDACTOR_MODEL=gemini-3.5-flash-lite
Use --estimar para ver cuántos turnos y llamadas aproximadas implica la selección, y --max-casos /
--categorias para correr por partes.

Uso (desde backend/):
    uv run python evaluacion/evaluar_agente.py --estimar
    uv run python evaluacion/evaluar_agente.py --pausa 6 --categorias saludo fuera_de_dominio voz_ruido
    uv run python evaluacion/evaluar_agente.py --pausa 8
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

# Llamadas aproximadas a Gemini por intención (analista + investigador + evaluador + redactor).
LLAMADAS_POR_TURNO_BUSQUEDA = 5
LLAMADAS_POR_TURNO_DIRECTO = 1


def cargar_casos(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def filtrar_casos(casos: list[dict], categorias: list[str] | None, maximo: int | None) -> list[dict]:
    if categorias:
        casos = [c for c in casos if c["categoria"] in categorias]
    return casos[:maximo] if maximo else casos


def estimar(casos: list[dict]) -> dict:
    """Estimación gruesa: los turnos previos de un caso suelen ser búsquedas; el último depende de la intención."""
    busca = ("consulta_sigem", "seguimiento")
    llamadas = sum(
        LLAMADAS_POR_TURNO_BUSQUEDA * (len(c["turnos"]) - 1)
        + (LLAMADAS_POR_TURNO_BUSQUEDA if c["intencion_esperada"] in busca else LLAMADAS_POR_TURNO_DIRECTO)
        for c in casos
    )
    return {"casos": len(casos), "turnos": sum(len(c["turnos"]) for c in casos), "llamadas_aprox": llamadas}


def evaluar_caso(agente, ejecutar_turno, caso: dict, pausa: float, sello: str) -> dict:
    thread_id = f"eval-{caso['id']}-{sello}"
    perfil = caso["perfil"]
    inicio = time.time()
    resultado = None
    for indice, turno in enumerate(caso["turnos"]):
        origen = turno.get("origen", "texto")
        etiquetas = [perfil, "evaluacion"] + (["voz"] if origen == "voz" else [])
        resultado = ejecutar_turno(agente, thread_id, turno["mensaje"], perfil, origen, etiquetas)
        if indice < len(caso["turnos"]) - 1 and pausa:
            time.sleep(pausa)

    analisis = resultado.get("analisis") or {}
    archivos_fuentes = {f.get("archivo") for f in resultado.get("fuentes", [])}
    archivos_candidatos = {c.get("archivo") for c in resultado.get("candidatos", [])}
    esperados = set(caso.get("archivos_esperados", []))
    espera_candidatos = bool(caso.get("candidatos_esperados"))

    metricas = {"intencion_ok": analisis.get("intencion") == caso["intencion_esperada"]}
    if esperados:
        encontrados = archivos_candidatos if espera_candidatos else archivos_fuentes
        metricas["archivos_ok"] = bool(esperados & encontrados)
    if espera_candidatos:
        metricas["candidatos_ok"] = bool(resultado.get("candidatos"))
    if caso["intencion_esperada"] == "no_comprensible":
        metricas["alerta_ok"] = bool(resultado.get("es_alerta"))

    return {
        "id": caso["id"],
        "categoria": caso["categoria"],
        "perfil": perfil,
        "mensaje": caso["turnos"][-1]["mensaje"],
        "intencion_esperada": caso["intencion_esperada"],
        "intencion_obtenida": analisis.get("intencion"),
        "archivos_esperados": sorted(esperados),
        "archivos_en_fuentes": sorted(a for a in archivos_fuentes if a),
        "archivos_en_candidatos": sorted(a for a in archivos_candidatos if a),
        "respuesta": resultado["messages"][-1].text,
        "pasos": resultado.get("pasos", []),
        "metricas": metricas,
        "tiempo_s": round(time.time() - inicio, 1),
    }


def es_cuota_diaria(error: Exception) -> bool:
    texto = str(error)
    return "RESOURCE_EXHAUSTED" in texto and ("PerDay" in texto or "per day" in texto.lower())


def resumir(resultados: list[dict]) -> dict:
    grupos = defaultdict(list)
    for resultado in resultados:
        grupos[resultado["categoria"]].append(resultado)

    def porcentaje(grupo: list[dict], clave: str):
        valores = [r["metricas"][clave] for r in grupo if clave in r.get("metricas", {})]
        return round(sum(valores) / len(valores), 3) if valores else None

    def resumen_grupo(grupo: list[dict]) -> dict:
        con_error = [r for r in grupo if "error" in r]
        sin_error = [r for r in grupo if "error" not in r]
        return {
            "casos": len(grupo),
            "errores": len(con_error),
            "intencion": porcentaje(sin_error, "intencion_ok"),
            "archivos": porcentaje(sin_error, "archivos_ok"),
            "candidatos": porcentaje(sin_error, "candidatos_ok"),
            "alerta": porcentaje(sin_error, "alerta_ok"),
            "tiempo_s_promedio": round(sum(r["tiempo_s"] for r in sin_error) / len(sin_error), 1) if sin_error else None,
        }

    resumen = {categoria: resumen_grupo(grupo) for categoria, grupo in sorted(grupos.items())}
    resumen["GLOBAL"] = resumen_grupo(resultados)
    return resumen


def mostrar_resumen(resumen: dict, linea_base: dict | None) -> None:
    def fmt(valor):
        return "  -  " if valor is None else f"{valor:5.2f}"

    print(f"\n{'categoría':<34}{'n':>3}{'err':>4}{'intención':>11}{'archivos':>10}{'candid.':>9}{'alerta':>8}{'seg':>7}{'base hit@8':>12}")
    for categoria, m in resumen.items():
        base = ""
        if linea_base and categoria in linea_base:
            base = fmt(linea_base[categoria].get("hit@8"))
        print(
            f"{categoria:<34}{m['casos']:>3}{m['errores']:>4}{fmt(m['intencion']):>11}{fmt(m['archivos']):>10}"
            f"{fmt(m['candidatos']):>9}{fmt(m['alerta']):>8}{(m['tiempo_s_promedio'] or 0):>7.1f}{base:>12}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(RAIZ_BACKEND / "evaluacion" / "dataset.jsonl"))
    parser.add_argument("--pausa", type=float, default=5.0, help="Segundos entre turnos y entre casos (cuota). Default 5.")
    parser.add_argument("--categorias", nargs="*", help="Solo estas categorías.")
    parser.add_argument("--max-casos", type=int, help="Máximo de casos a correr (los primeros tras filtrar).")
    parser.add_argument("--estimar", action="store_true", help="Solo muestra cuántos turnos/llamadas implica; no llama a Gemini.")
    parser.add_argument("--salida", help="Archivo de resultados (default: evaluacion/resultados_<fecha>.json).")
    args = parser.parse_args()

    casos = filtrar_casos(cargar_casos(Path(args.dataset)), args.categorias, args.max_casos)
    if not casos:
        print("No hay casos para esa selección.")
        return 2

    if args.estimar:
        e = estimar(casos)
        print(f"{e['casos']} casos, {e['turnos']} turnos, ~{e['llamadas_aprox']} llamadas a Gemini (estimación gruesa).")
        return 0

    # Memoria en proceso: un thread por caso, sin Postgres. Debe fijarse antes de importar la config.
    os.environ["CHECKPOINTER"] = "memory"
    os.chdir(RAIZ_BACKEND)
    sys.path.insert(0, str(RAIZ_BACKEND))

    from agente_sigem.config import config

    if not config.GOOGLE_API_KEY:
        print("ERROR: falta GOOGLE_API_KEY. No se evaluó nada.")
        return 2
    if not (Path(config.CHROMA_PERSIST_DIR) / "chroma.sqlite3").exists():
        print(f"ERROR: no hay un Chroma local en {Path(config.CHROMA_PERSIST_DIR).resolve()}. No se evaluó nada.")
        return 2

    from agente_sigem import reranker
    from agente_sigem.agent import construir_grafo
    from agente_sigem.api import ejecutar_turno
    from agente_sigem.checkpointer import crear_checkpointer

    checkpointer, cerrar = crear_checkpointer()
    agente = construir_grafo(checkpointer)
    if config.RERANKER_HABILITADO:
        print(f"Reranker precargado: {reranker.precargar()}")

    print(
        f"Evaluando {len(casos)} casos | modelos: analista={config.ANALISTA_MODEL} "
        f"investigador={config.INVESTIGADOR_MODEL} evaluador={config.EVALUADOR_MODEL} redactor={config.REDACTOR_MODEL}"
    )
    sello = datetime.now().strftime("%Y%m%d%H%M%S")
    resultados = []
    abortado = None
    for caso in casos:
        try:
            resultado = evaluar_caso(agente, ejecutar_turno, caso, args.pausa, sello)
            marcas = " ".join(f"{k.replace('_ok', '')}={'OK' if v else 'NO'}" for k, v in resultado["metricas"].items())
            print(f"  {caso['id']:<14} {resultado['tiempo_s']:>5.1f}s  {marcas}  ({resultado['intencion_obtenida']})")
        except Exception as error:
            resultado = {
                "id": caso["id"], "categoria": caso["categoria"], "perfil": caso["perfil"],
                "mensaje": caso["turnos"][-1]["mensaje"], "error": str(error)[:500], "metricas": {}, "tiempo_s": 0.0,
            }
            print(f"  {caso['id']:<14} ERROR: {str(error)[:140]}")
            if es_cuota_diaria(error):
                resultados.append(resultado)
                abortado = "Cuota diaria de Gemini agotada: se detuvo la corrida (resultados parciales guardados)."
                break
        resultados.append(resultado)
        if args.pausa:
            time.sleep(args.pausa)
    cerrar()

    resumen = resumir(resultados)
    linea_base = None
    ruta_base = RAIZ_BACKEND / "evaluacion" / "linea_base_recuperacion.json"
    if ruta_base.exists():
        linea_base = json.loads(ruta_base.read_text(encoding="utf-8")).get("resumen")
    mostrar_resumen(resumen, linea_base)
    if abortado:
        print(f"\n{abortado}")

    salida = Path(args.salida) if args.salida else RAIZ_BACKEND / "evaluacion" / f"resultados_{sello}.json"
    salida.write_text(
        json.dumps(
            {
                "fecha": datetime.now().isoformat(timespec="seconds"),
                "modelos": {
                    "analista": config.ANALISTA_MODEL, "investigador": config.INVESTIGADOR_MODEL,
                    "evaluador": config.EVALUADOR_MODEL, "redactor": config.REDACTOR_MODEL,
                },
                "reranker": config.RERANKER_HABILITADO,
                "pausa_s": args.pausa,
                "abortado": abortado,
                "resumen": resumen,
                "casos": resultados,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nGuardado en {salida}")
    return 1 if abortado else 0


if __name__ == "__main__":
    raise SystemExit(main())
