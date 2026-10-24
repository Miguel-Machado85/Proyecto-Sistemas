"""Evaluación de extremo a extremo: corre el grafo completo sobre el dataset (sección 13).

Por cada caso crea un thread nuevo y ejecuta todos sus turnos en orden (CHECKPOINTER=memory), de modo
que los casos de seguimiento y de búsqueda global con documento activo prueban la memoria real.
Mide sobre el ÚLTIMO turno de cada caso:
  - intencion_ok:  la intención del analista coincide con `intencion_esperada`.
  - archivos_ok:   algún archivo de `archivos_esperados` aparece en `fuentes` (o en `candidatos`
                   cuando el caso espera candidatos).
  - candidatos_ok: hay candidatos cuando el caso los espera (`candidatos_esperados`).
  - alerta_ok:     `es_alerta` es True cuando se espera `no_comprensible`.
  - solicita_archivo_ok / incluye_contenido_ok: el analista marcó `solicita_archivo` / `incluye_contenido` como
                   dice `solicita_archivo_esperado` / `incluye_contenido_esperado` (solo en casos que lo definen).
  - boton_ok:      la respuesta no menciona un "botón" de descarga cuando el usuario no recibe ningún archivo.
  - entrega_ok:    con `solicita_archivo_esperado`, el archivo esperado llegó en `archivos` (lo que ve el usuario) si se
                   esperaba entrega, o `archivos` quedó vacío si no.

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
import re
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


def filtrar_casos(
    casos: list[dict], categorias: list[str] | None, maximo: int | None, ids: list[str] | None = None
) -> list[dict]:
    if ids:
        casos = [c for c in casos if c["id"] in ids]
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
    # Lo que el usuario recibe como botón: misma compuerta que la API (solo si el analista marcó solicita_archivo).
    archivos_entregados = {a.get("archivo") for a in resultado.get("archivos", [])} if analisis.get("solicita_archivo") else set()
    esperados = set(caso.get("archivos_esperados", []))
    espera_candidatos = bool(caso.get("candidatos_esperados"))

    metricas = {"intencion_ok": analisis.get("intencion") == caso["intencion_esperada"]}
    if esperados:
        encontrados = archivos_candidatos if espera_candidatos else archivos_fuentes | archivos_entregados
        metricas["archivos_ok"] = bool(esperados & encontrados)
    if espera_candidatos:
        metricas["candidatos_ok"] = bool(resultado.get("candidatos"))
    if caso["intencion_esperada"] == "no_comprensible":
        metricas["alerta_ok"] = bool(resultado.get("es_alerta"))
    if "solicita_archivo_esperado" in caso:
        espera_archivo = bool(caso["solicita_archivo_esperado"])
        metricas["solicita_archivo_ok"] = bool(analisis.get("solicita_archivo")) == espera_archivo
        if "incluye_contenido_esperado" in caso:
            metricas["incluye_contenido_ok"] = bool(analisis.get("incluye_contenido")) == bool(caso["incluye_contenido_esperado"])
        metricas["entrega_ok"] = bool(esperados & archivos_entregados) if espera_archivo else not archivos_entregados

    # La respuesta no puede prometer un botón de descarga que la API no va a mostrar.
    metricas["boton_ok"] = not menciona_boton_inexistente(resultado["messages"][-1].text, archivos_entregados)

    sin_informacion = respuesta_sin_informacion(resultado)
    if caso["categoria"] == "referencia_inexistente":
        # Documento que no existe: debe decirlo y no mostrar fuentes de relleno.
        metricas["negativa_ok"] = sin_informacion and not resultado.get("fuentes")
    elif esperados and not espera_candidatos:
        # Documento que sí existe: la respuesta no puede decir que no hay información.
        metricas["informacion_ok"] = not sin_informacion

    return {
        "id": caso["id"],
        "categoria": caso["categoria"],
        "perfil": perfil,
        "razonamiento_analista": analisis.get("razonamiento"),
        "degradado": motivos_de_degradacion(resultado),
        "mensaje": caso["turnos"][-1]["mensaje"],
        "intencion_esperada": caso["intencion_esperada"],
        "intencion_obtenida": analisis.get("intencion"),
        "archivos_esperados": sorted(esperados),
        "archivos_en_fuentes": sorted(a for a in archivos_fuentes if a),
        "archivos_en_candidatos": sorted(a for a in archivos_candidatos if a),
        "archivos_entregados": sorted(a for a in archivos_entregados if a),
        "solicita_archivo_obtenido": bool(analisis.get("solicita_archivo")),
        "incluye_contenido_obtenido": bool(analisis.get("incluye_contenido")),
        "respuesta": resultado["messages"][-1].text,
        "pasos": resultado.get("pasos", []),
        "metricas": metricas,
        "tiempo_s": round(time.time() - inicio, 1),
    }


FRASES_NEGATIVAS = (
    "no se encuentra", "no encuentro", "no hay información", "no hay informacion", "no está disponible",
    "no esta disponible", "no cuento con", "no aparece", "no contienen", "no se encontr", "no existe",
)
MARCA_SIN_INFORMACION = "sin información pertinente"
# Con límite de palabra al inicio: "humano se encuentra" no debe contar como "no se encuentra".
PATRON_FRASE_NEGATIVA = re.compile("|".join(r"(?<!\w)" + re.escape(frase) for frase in FRASES_NEGATIVAS))
PATRON_BOTON = re.compile(r"(?<!\w)bot[oó]n(?:es)?(?!\w)", re.IGNORECASE)


def respuesta_sin_informacion(resultado: dict) -> bool:
    """True si el redactor lo declaró (paso del redactor) o la respuesta lo dice con frases típicas."""
    if any(
        p.get("agente") == "redactor" and MARCA_SIN_INFORMACION in p.get("accion", "")
        for p in resultado.get("pasos", [])
    ):
        return True
    texto = resultado["messages"][-1].text.lower()
    return bool(PATRON_FRASE_NEGATIVA.search(texto))


def menciona_boton_inexistente(texto: str, archivos_entregados: set) -> bool:
    """True si la respuesta habla de un botón de descarga y el usuario no recibe ningún archivo."""
    return bool(PATRON_BOTON.search(texto)) and not archivos_entregados


def es_cuota_diaria(error: Exception) -> bool:
    texto = str(error)
    return "RESOURCE_EXHAUSTED" in texto and ("PerDay" in texto or "per day" in texto.lower())


def es_cuota_por_minuto(error: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(error) and not es_cuota_diaria(error)


def motivos_de_degradacion(resultado: dict) -> list[str]:
    """Señales de que algún agente cayó en su modo fail-open: la medición no representa al sistema sano."""
    motivos = []
    if str((resultado.get("analisis") or {}).get("razonamiento", "")).startswith("Análisis no disponible"):
        motivos.append("analista_fail_open")
    if any(
        p.get("agente") == "evaluador" and p.get("accion", "").startswith("Evaluación no disponible")
        for p in resultado.get("pasos", [])
    ):
        motivos.append("evaluador_fail_open")
    return motivos


def evaluar_con_reintentos(
    agente, ejecutar_turno, caso: dict, pausa: float, sello: str, espera_cuota: float = 65.0, reintentos: int = 2
) -> dict:
    """Reintenta el caso (esperando a que se renueve la cuota por minuto) si falló o quedó degradado."""
    for intento in range(reintentos + 1):
        try:
            resultado = evaluar_caso(agente, ejecutar_turno, caso, pausa, f"{sello}-{intento}")
        except Exception as error:
            if es_cuota_por_minuto(error) and intento < reintentos:
                print(f"    cuota por minuto agotada en {caso['id']}: espero {espera_cuota:.0f}s y reintento", flush=True)
                time.sleep(espera_cuota)
                continue
            raise
        if resultado["degradado"] and intento < reintentos:
            print(f"    {caso['id']} degradado ({', '.join(resultado['degradado'])}): espero {espera_cuota:.0f}s y reintento", flush=True)
            time.sleep(espera_cuota)
            continue
        resultado["intentos"] = intento + 1
        return resultado


def resumir(resultados: list[dict]) -> dict:
    grupos = defaultdict(list)
    for resultado in resultados:
        grupos[resultado["categoria"]].append(resultado)

    def porcentaje(grupo: list[dict], clave: str):
        valores = [r["metricas"][clave] for r in grupo if clave in r.get("metricas", {})]
        return round(sum(valores) / len(valores), 3) if valores else None

    def resumen_grupo(grupo: list[dict]) -> dict:
        con_error = [r for r in grupo if "error" in r]
        degradados = [r for r in grupo if r.get("degradado")]
        sin_error = [r for r in grupo if "error" not in r and not r.get("degradado")]
        return {
            "casos": len(grupo),
            "errores": len(con_error),
            "degradados": len(degradados),
            "intencion": porcentaje(sin_error, "intencion_ok"),
            "archivos": porcentaje(sin_error, "archivos_ok"),
            "candidatos": porcentaje(sin_error, "candidatos_ok"),
            "alerta": porcentaje(sin_error, "alerta_ok"),
            "solicita_archivo": porcentaje(sin_error, "solicita_archivo_ok"),
            "entrega": porcentaje(sin_error, "entrega_ok"),
            "informacion": porcentaje(sin_error, "informacion_ok"),
            "negativa": porcentaje(sin_error, "negativa_ok"),
            "boton": porcentaje(sin_error, "boton_ok"),
            "tiempo_s_promedio": round(sum(r["tiempo_s"] for r in sin_error) / len(sin_error), 1) if sin_error else None,
        }

    resumen = {categoria: resumen_grupo(grupo) for categoria, grupo in sorted(grupos.items())}
    resumen["GLOBAL"] = resumen_grupo(resultados)
    return resumen


def mostrar_resumen(resumen: dict, linea_base: dict | None) -> None:
    def fmt(valor):
        return "  -  " if valor is None else f"{valor:5.2f}"

    print(f"\n{'categoría':<34}{'n':>3}{'err':>4}{'degr':>5}{'intención':>11}{'archivos':>10}{'candid.':>9}{'alerta':>8}{'info':>6}{'neg.':>6}{'sol.arch':>9}{'entrega':>8}{'boton':>7}{'seg':>7}{'base hit@8':>12}")
    for categoria, m in resumen.items():
        base = ""
        if linea_base and categoria in linea_base:
            base = fmt(linea_base[categoria].get("hit@8"))
        print(
            f"{categoria:<34}{m['casos']:>3}{m['errores']:>4}{m['degradados']:>5}{fmt(m['intencion']):>11}{fmt(m['archivos']):>10}"
            f"{fmt(m['candidatos']):>9}{fmt(m['alerta']):>8}{fmt(m['informacion']):>6}{fmt(m['negativa']):>6}{fmt(m['solicita_archivo']):>9}{fmt(m['entrega']):>8}{fmt(m['boton']):>7}{(m['tiempo_s_promedio'] or 0):>7.1f}{base:>12}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(RAIZ_BACKEND / "evaluacion" / "dataset.jsonl"))
    parser.add_argument("--pausa", type=float, default=5.0, help="Segundos entre turnos y entre casos (cuota). Default 5.")
    parser.add_argument("--categorias", nargs="*", help="Solo estas categorías.")
    parser.add_argument("--ids", nargs="*", help="Solo estos casos (p. ej. inv-04 inex-01).")
    parser.add_argument("--max-casos", type=int, help="Máximo de casos a correr (los primeros tras filtrar).")
    parser.add_argument("--estimar", action="store_true", help="Solo muestra cuántos turnos/llamadas implica; no llama a Gemini.")
    parser.add_argument("--salida", help="Archivo de resultados (default: evaluacion/resultados_<fecha>.json).")
    args = parser.parse_args()

    casos = filtrar_casos(cargar_casos(Path(args.dataset)), args.categorias, args.max_casos, args.ids)
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
    # Con CHROMA_MODO=http el índice vive en el servidor: no hace falta la carpeta local.
    if config.CHROMA_MODO == "local" and not (Path(config.CHROMA_PERSIST_DIR) / "chroma.sqlite3").exists():
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
            resultado = evaluar_con_reintentos(agente, ejecutar_turno, caso, args.pausa, sello)
            marcas = " ".join(f"{k.replace('_ok', '')}={'OK' if v else 'NO'}" for k, v in resultado["metricas"].items())
            aviso = f"  [DEGRADADO: {', '.join(resultado['degradado'])}]" if resultado["degradado"] else ""
            print(f"  {caso['id']:<14} {resultado['tiempo_s']:>5.1f}s  {marcas}  ({resultado['intencion_obtenida']}){aviso}", flush=True)
        except Exception as error:
            resultado = {
                "id": caso["id"], "categoria": caso["categoria"], "perfil": caso["perfil"],
                "mensaje": caso["turnos"][-1]["mensaje"], "error": str(error)[:500], "metricas": {}, "tiempo_s": 0.0,
            }
            print(f"  {caso['id']:<14} ERROR: {str(error)[:140]}", flush=True)
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
