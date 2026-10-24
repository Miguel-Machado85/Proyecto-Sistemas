"""Enlaza los documentos indexados en Chroma con su copia en el repositorio de archivos (Drive).

1. Lista el repositorio (carpeta de Drive) y el catálogo de Chroma (`nombre_archivo`).
2. Empareja por nombre exacto y, si no, por nombre normalizado (sin tildes, mayúsculas, signos ni
   extensión). Los dudosos (varios candidatos, o un archivo de Drive pedido por dos documentos) se
   reportan y NO se enlazan.
3. Por defecto es una SIMULACIÓN: solo muestra el reporte. `--aplicar` escribe la tabla `archivos_documento`.
4. Reporta tres listas: enlazados, en Chroma pero no en Drive (no se podrán descargar) y en Drive pero no
   indexados. Con `--subir-faltantes DIR` (y `--aplicar`) sube los faltantes desde una carpeta local; sin
   `--aplicar` solo informa cuáles subiría. Subir requiere permiso de Editor sobre la carpeta.

Uso (desde backend/):
    uv run python scripts/sincronizar_drive.py
    uv run python scripts/sincronizar_drive.py --aplicar
    uv run python scripts/sincronizar_drive.py --aplicar --subir-faltantes ./documentos
"""

import argparse
import mimetypes
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

RAIZ_BACKEND = Path(__file__).resolve().parent.parent
if str(RAIZ_BACKEND) not in sys.path:
    sys.path.insert(0, str(RAIZ_BACKEND))

from agente_sigem.repositorio_archivos import ArchivoRemoto  # noqa: E402


def clave_normalizada(nombre: str) -> str:
    """Nombre sin extensión, sin tildes, en minúsculas y solo con letras y dígitos."""
    sin_extension = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", nombre)
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", sin_extension.lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", sin_tildes)


@dataclass
class Plan:
    enlazados: list[tuple[str, ArchivoRemoto, str]] = field(default_factory=list)  # (archivo, remoto, criterio)
    dudosos: list[tuple[str, list[ArchivoRemoto]]] = field(default_factory=list)
    sin_drive: list[str] = field(default_factory=list)
    huerfanos: list[ArchivoRemoto] = field(default_factory=list)


def planificar(catalogo: dict[str, str | None], remotos: list[ArchivoRemoto]) -> Plan:
    """`catalogo` es {nombre_archivo: titulo}. No escribe nada."""
    plan = Plan()
    por_nombre: dict[str, list[ArchivoRemoto]] = {}
    por_clave: dict[str, list[ArchivoRemoto]] = {}
    for remoto in remotos:
        por_nombre.setdefault(remoto.nombre, []).append(remoto)
        por_clave.setdefault(clave_normalizada(remoto.nombre), []).append(remoto)

    propuestas: dict[str, tuple[ArchivoRemoto, str]] = {}
    for archivo in sorted(catalogo):
        exactos = por_nombre.get(archivo, [])
        candidatos, criterio = (exactos, "nombre exacto") if exactos else (por_clave.get(clave_normalizada(archivo), []), "nombre normalizado")
        if len(candidatos) == 1:
            propuestas[archivo] = (candidatos[0], criterio)
        elif len(candidatos) > 1:
            plan.dudosos.append((archivo, candidatos))
        else:
            plan.sin_drive.append(archivo)

    # Un mismo archivo de Drive pedido por dos documentos del catálogo es dudoso: no se enlaza ninguno.
    pedidos: dict[str, list[str]] = {}
    for archivo, (remoto, _) in propuestas.items():
        pedidos.setdefault(remoto.id_externo, []).append(archivo)
    for archivo, (remoto, criterio) in propuestas.items():
        if len(pedidos[remoto.id_externo]) > 1:
            plan.dudosos.append((archivo, [remoto]))
        else:
            plan.enlazados.append((archivo, remoto, criterio))

    usados = {remoto.id_externo for _, remoto, _ in plan.enlazados}
    dudosos_ids = {r.id_externo for _, candidatos in plan.dudosos for r in candidatos}
    plan.huerfanos = [r for r in remotos if r.id_externo not in usados and r.id_externo not in dudosos_ids]
    return plan


def aplicar(plan: Plan, catalogo: dict[str, str | None], proveedor: str, registrar, repositorio=None, carpeta_faltantes: Path | None = None) -> dict:
    """Escribe los enlaces y, si se pidió, sube los faltantes. `registrar` tiene la firma de db.registrar_archivo."""
    resultado = {"enlazados": 0, "subidos": [], "no_encontrados_localmente": [], "errores": []}
    for archivo, remoto, _ in plan.enlazados:
        registrar(nombre_archivo=archivo, proveedor=proveedor, id_externo=remoto.id_externo,
                  titulo=catalogo.get(archivo), mime_type=remoto.mime_type, tamano_bytes=remoto.tamano_bytes)
        resultado["enlazados"] += 1
    if carpeta_faltantes is not None and repositorio is not None:
        for archivo in plan.sin_drive:
            ruta = carpeta_faltantes / archivo
            if not ruta.is_file():
                resultado["no_encontrados_localmente"].append(archivo)
                continue
            mime = mimetypes.guess_type(archivo)[0] or "application/octet-stream"
            try:
                id_externo = repositorio.subir(ruta, archivo, mime)
            except Exception as error:  # ErrorRepositorio: se informa y se sigue con el resto
                resultado["errores"].append((archivo, str(error)))
                continue
            registrar(nombre_archivo=archivo, proveedor=proveedor, id_externo=id_externo,
                      titulo=catalogo.get(archivo), mime_type=mime, tamano_bytes=ruta.stat().st_size)
            resultado["subidos"].append(archivo)
    return resultado


def imprimir_reporte(plan: Plan, subir_desde: Path | None = None) -> None:
    print(f"\nEnlazados ({len(plan.enlazados)}):")
    for archivo, remoto, criterio in plan.enlazados:
        marca = "" if criterio == "nombre exacto" else f"  [{criterio}: Drive='{remoto.nombre}']"
        print(f"  ✓ {archivo}{marca}")
    print(f"\nDudosos, NO se enlazan ({len(plan.dudosos)}):")
    for archivo, candidatos in plan.dudosos:
        print(f"  ? {archivo} -> {[c.nombre for c in candidatos]}")
    print(f"\nEn Chroma pero NO en Drive ({len(plan.sin_drive)}) — no se podrán descargar:")
    for archivo in plan.sin_drive:
        extra = ""
        if subir_desde is not None:
            extra = "  (se subiría)" if (subir_desde / archivo).is_file() else "  (no está en la carpeta local)"
        print(f"  ✗ {archivo}{extra}")
    print(f"\nEn Drive pero NO indexados ({len(plan.huerfanos)}):")
    for remoto in plan.huerfanos:
        print(f"  · {remoto.nombre}")


def main() -> int:
    from agente_sigem import db
    from agente_sigem.config import config
    from agente_sigem.repositorio_archivos import ErrorRepositorio, obtener_repositorio
    from agente_sigem.retrieval import obtener_catalogo_documentos

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aplicar", action="store_true", help="Escribe la tabla archivos_documento (y sube si hay --subir-faltantes).")
    parser.add_argument("--subir-faltantes", metavar="DIR", type=Path, help="Carpeta local con los PDFs que faltan en el repositorio.")
    args = parser.parse_args()

    try:
        repositorio = obtener_repositorio()
        remotos = repositorio.listar()
    except ErrorRepositorio as error:
        print(f"ERROR: {error}")
        return 1
    catalogo = {d.archivo: d.titulo for d in obtener_catalogo_documentos().documentos}
    print(f"Repositorio: {repositorio.proveedor} · {len(remotos)} archivos · catálogo de Chroma: {len(catalogo)} documentos")

    plan = planificar(catalogo, remotos)
    imprimir_reporte(plan, args.subir_faltantes)

    if not args.aplicar:
        print("\nSIMULACIÓN: no se escribió nada. Usa --aplicar para guardar los enlaces.")
        return 0
    db.init_db()
    resultado = aplicar(plan, catalogo, repositorio.proveedor, db.registrar_archivo, repositorio, args.subir_faltantes)
    print(f"\nAPLICADO: {resultado['enlazados']} enlazados, {len(resultado['subidos'])} subidos.")
    for archivo, error in resultado["errores"]:
        print(f"  ERROR al subir {archivo}: {error}")
    for archivo in resultado["no_encontrados_localmente"]:
        print(f"  Sin archivo local para subir: {archivo}")
    return 1 if resultado["errores"] else 0


if __name__ == "__main__":
    sys.exit(main())
