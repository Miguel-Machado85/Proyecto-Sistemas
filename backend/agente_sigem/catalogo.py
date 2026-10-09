"""Catálogo normalizado para resolver referencias a documentos SIGEM."""

from collections import defaultdict
from dataclasses import dataclass
import re
import unicodedata


TIPOS_DOCUMENTO = "decreto|resolucion|acuerdo|ley|formato"
PATRON_REFERENCIA = re.compile(
    rf"\b(?P<tipo>{TIPOS_DOCUMENTO})\s*"
    r"(?:(?:(?:n(?:o|ro)?|numero)\.?\s*|#\s*))?"
    r"(?P<numero>\d{1,5})"
    r"(?:\s*(?:de|/|-)\s*(?P<anio>(?:19|20)\d{2}))?\b",
    re.IGNORECASE,
)
PATRON_CODIGO_ARCHIVO = re.compile(
    r"\b(?:documento|archivo|codigo)\s*(?:no\.?\s*)?(?P<codigo>\d{3,8})\b",
    re.IGNORECASE,
)
PATRON_ANIO = re.compile(r"\b((?:19|20)\d{2})\b")


def normalizar(texto: str) -> str:
    """Uniforma acentos y variantes tipográficas para comparar referencias."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(caracter for caracter in texto if unicodedata.category(caracter) != "Mn")
    texto = texto.replace("n.º", "no").replace("nº", "no").replace("n°", "no").replace("n.o", "no")
    return texto.replace("º", "o").replace("°", "o").replace("#", " # ")


def tipo_canonico(tipo: str) -> str:
    return normalizar(tipo).replace("resolucion", "resolucion")


def extraer_anio(texto: str | None) -> int | None:
    if not texto:
        return None
    coincidencia = PATRON_ANIO.search(texto)
    return int(coincidencia.group(1)) if coincidencia else None


@dataclass(frozen=True)
class ReferenciaDocumento:
    tipo: str
    numero: int
    anio: int | None


def extraer_referencia(consulta: str) -> ReferenciaDocumento | None:
    match = PATRON_REFERENCIA.search(normalizar(consulta))
    if not match:
        return None
    return ReferenciaDocumento(
        tipo=tipo_canonico(match.group("tipo")),
        numero=int(match.group("numero")),
        anio=int(match.group("anio")) if match.group("anio") else None,
    )


@dataclass(frozen=True)
class DocumentoCatalogado:
    archivo: str
    titulo: str
    descripcion: str | None
    tipo: str | None
    numero: int | None
    anio: int | None
    codigo_archivo: str | None
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ResolucionCatalogo:
    estado: str
    documentos: tuple[DocumentoCatalogado, ...]


class CatalogoDocumentos:
    """Representación deduplicada de los documentos de una colección Chroma."""

    def __init__(self, documentos: list[DocumentoCatalogado]):
        self.documentos = tuple(documentos)

    @classmethod
    def desde_metadatas(cls, metadatas: list[dict]) -> "CatalogoDocumentos":
        grupos = defaultdict(list)
        for metadata in metadatas:
            metadata = metadata or {}
            archivo = str(metadata.get("nombre_archivo") or metadata.get("source") or "")
            if archivo:
                grupos[archivo].append(metadata)

        documentos = []
        for archivo, fragmentos in grupos.items():
            titulo = str(next((m.get("titulo") for m in fragmentos if m.get("titulo")), archivo))
            descripcion = next((m.get("descripcion") for m in fragmentos if m.get("descripcion")), None)
            referencia = extraer_referencia(titulo)
            fecha = next((m.get("fecha_expedicion") for m in fragmentos if m.get("fecha_expedicion")), None)
            codigo = re.match(r"(?P<codigo>\d{3,8})_", archivo)

            tipo = referencia.tipo if referencia else None
            numero = referencia.numero if referencia else None
            anio = referencia.anio if referencia and referencia.anio else extraer_anio(str(fecha))
            aliases = cls._crear_aliases(titulo, archivo, tipo, numero, anio)
            documentos.append(
                DocumentoCatalogado(
                    archivo=archivo,
                    titulo=titulo,
                    descripcion=descripcion,
                    tipo=tipo,
                    numero=numero,
                    anio=anio,
                    codigo_archivo=codigo.group("codigo") if codigo else None,
                    aliases=aliases,
                )
            )
        return cls(documentos)

    @staticmethod
    def _crear_aliases(
        titulo: str, archivo: str, tipo: str | None, numero: int | None, anio: int | None
    ) -> tuple[str, ...]:
        aliases = {normalizar(titulo), normalizar(archivo)}
        if tipo and numero is not None:
            aliases.update({f"{tipo} {numero}", f"{tipo} no {numero}", f"{tipo} numero {numero}"})
            if anio:
                aliases.update({f"{tipo} {numero} de {anio}", f"{tipo} {numero}/{anio}", f"{tipo} {numero}-{anio}"})
        return tuple(sorted(aliases))

    def resolver(self, consulta: str) -> ResolucionCatalogo:
        referencia = extraer_referencia(consulta)
        if referencia:
            candidatos = [
                documento
                for documento in self.documentos
                if documento.tipo == referencia.tipo
                and documento.numero == referencia.numero
                and (referencia.anio is None or documento.anio == referencia.anio)
            ]
            return self._resultado(candidatos)

        coincidencia_codigo = PATRON_CODIGO_ARCHIVO.search(normalizar(consulta))
        if coincidencia_codigo:
            candidatos = [
                documento for documento in self.documentos if documento.codigo_archivo == coincidencia_codigo.group("codigo")
            ]
            return self._resultado(candidatos)

        return ResolucionCatalogo(estado="sin_coincidencia", documentos=())

    def sugerencias(self, consulta: str, limite: int = 5) -> list[DocumentoCatalogado]:
        termino = normalizar(consulta)
        if len(termino) < 3:
            return []
        palabras = set(re.findall(r"[a-z0-9]+", termino))
        candidatos = []
        for documento in self.documentos:
            titulo = normalizar(documento.titulo)
            descripcion = normalizar(documento.descripcion or "")
            puntaje = 0
            puntaje += 100 if termino in documento.aliases else 0
            puntaje += 60 if termino in titulo else 0
            puntaje += 20 if termino in descripcion else 0
            puntaje += sum(5 for palabra in palabras if palabra in titulo or palabra in descripcion)
            if puntaje:
                candidatos.append((documento, puntaje))
        return [documento for documento, _ in sorted(candidatos, key=lambda item: item[1], reverse=True)[:limite]]

    @staticmethod
    def _resultado(candidatos: list[DocumentoCatalogado]) -> ResolucionCatalogo:
        if len(candidatos) == 1:
            return ResolucionCatalogo(estado="exacta", documentos=tuple(candidatos))
        if candidatos:
            return ResolucionCatalogo(estado="ambigua", documentos=tuple(candidatos))
        return ResolucionCatalogo(estado="sin_coincidencia", documentos=())
