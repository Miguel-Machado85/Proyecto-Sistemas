# Actualización para la wiki (`home.md`)

Secciones nuevas o cambiadas por la etapa "entrega de documentos, Google Drive y Chroma como servicio". Cada bloque indica si **reemplaza** una sección
existente de la wiki o se **agrega** como nueva.

---

## [REEMPLAZA] Docker Compose

`Compose.yml` levanta cuatro servicios:

| Servicio | Imagen / build | Puertos | Notas |
|---|---|---|---|
| `backend` | `./backend` | `8001` | FastAPI + LangGraph. `CHROMA_MODO=http`, `CHROMA_HOST=chroma`, `CHECKPOINTER=postgres`. Monta `backend/documentos`, `backend/secrets` (**solo lectura**) y el volumen `hf-cache`. Espera a que `db` y `chroma` estén sanos. |
| `chroma` | `chromadb/chroma:1.5.9` | `127.0.0.1:8000` (solo depuración y migración) | Volumen `chroma-data` en `/data`. Healthcheck con `/dev/tcp` de bash (la imagen no trae `curl` ni Python). |
| `db` | `postgres:16-alpine` | `55432` | Historial de turnos, checkpointer del grafo y tabla `archivos_documento`. |
| `frontend` | `./frontend` (nginx) | `8080` | `index.html` (chat) y `admin.html` (carga administrativa). |

```bash
cp backend/.env.example backend/.env
docker compose -f Compose.yml up -d --build
```

La versión del servidor Chroma debe ser **la misma** que la del paquete `chromadb` del backend (hoy `1.5.9`); si se actualiza uno, se actualiza el otro.
El backend ya **no** monta `backend/chroma_db`.

---

## [REEMPLAZA] Chroma (ya no es embebido)

Antes, el backend abría la carpeta `backend/chroma_db` dentro de su propio proceso. Ahora Chroma es **un servicio aparte** y el backend se conecta por red
(`chromadb.HttpClient`). Ventajas: escala independiente de la API (más recursos, varias réplicas del backend comparten el mismo índice), la ingesta y las consultas
ya no compiten por un archivo SQLite, y mover Chroma a otra máquina es cambiar `CHROMA_HOST`.

| Variable | Valor | Descripción |
|---|---|---|
| `CHROMA_MODO` | `local` (por defecto) o `http` | `local`: carpeta `CHROMA_PERSIST_DIR` (tests, desarrollo y origen de la migración). `http`: servidor. **Compose usa `http`.** |
| `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_SSL` | `chroma`, `8000`, `false` en Compose | Dirección del servidor. |
| `COLLECTION_NAME`, `EMBEDDING_MODEL` | `langchain`, `models/gemini-embedding-2` | **No cambiar**: el índice se generó con ellos. |

**Migración sin regenerar embeddings** (`backend/scripts/migrar_chroma.py`): lee `chroma_db` en páginas de 500 (sobre una copia temporal; el original no se toca),
crea la colección destino con la misma configuración de distancia (l2), escribe con `upsert` (idempotente, se puede repetir si se corta) y verifica conteo, ids y el top-8 de
10 fragmentos al azar usando su propio embedding guardado como consulta (no gasta cuota de Gemini). El índice HNSW es aproximado, así que el top-8 de origen y destino puede
diferir en vecinos casi empatados; por eso la verificación compara cada lado con el top-8 exacto y exige que el destino no recupere peor que el origen.

```bash
docker compose -f Compose.yml up -d chroma
cd backend && CHROMA_MODO=http uv run python scripts/migrar_chroma.py
```

`GET /health` informa también el estado de Chroma (`{"status":"ok","chroma":"ok"}`) y responde `503` si el servidor no contesta.

**Catálogo con varias réplicas:** el catálogo de documentos (que usan `buscar_documento_por_referencia`, `listar_documentos` y `preparar_descarga`) guarda el
`collection.count()` con el que se construyó y se reconstruye si cambia, por lo que un documento indexado por otro proceso aparece sin reiniciar.

---

## [AGREGA] Entrega de documentos: Google Drive y endpoint de descarga

El agente puede **entregar el PDF original** de un documento. Los originales viven en una carpeta de Google Drive (o en disco en desarrollo) detrás de
`agente_sigem/repositorio_archivos.py`; la tabla `archivos_documento` de Postgres enlaza el `nombre_archivo` de Chroma con el `id_externo` del archivo.

- `GET /documentos/{archivo}/descargar` (el backend descarga de Drive y reenvía el archivo con `Content-Disposition: attachment` y el nombre original; la carpeta
  puede seguir privada). `DRIVE_MODO_ENTREGA=redirect` responde `302` al enlace de Drive. `404` amable si el documento no está en el catálogo o no tiene original;
  `503` si Drive no está disponible. El parámetro **nunca** se usa como ruta de disco: solo se compara con el catálogo.
- `ARCHIVOS_BACKEND=drive|local`, `GOOGLE_DRIVE_FOLDER_ID` (acepta ID o URL), `GOOGLE_OAUTH_CLIENT_FILE`, `GOOGLE_OAUTH_TOKEN_FILE`, `DRIVE_MODO_ENTREGA`.
- `retirar()` mueve a la papelera de Drive; el código **nunca borra de forma permanente**.
- Si el token falta, está dañado o caducó, la API arranca igual y solo la descarga responde `503`.

### Acceso a la carpeta de Drive (configuración definitiva)
1. Compartir la carpeta **por correo** con la cuenta de Google que usa el backend, con rol **Editor** (Lector basta para descargar; Editor es necesario para que la ingesta suba).
2. Dejar el acceso general del enlace en **Lector** o restringido. **No** usar "cualquiera con el enlace puede editar".
3. Los archivos que suba el backend pertenecen a esa cuenta (ocupan su espacio). A largo plazo conviene que la carpeta sea de una cuenta del proyecto.

### Cómo generar las credenciales de Drive (paso a paso, una sola vez)
1. **Google Cloud Console** → crear o elegir un proyecto (puede ser otro distinto al de la `GOOGLE_API_KEY` de Gemini) → **APIs y servicios → Biblioteca → Google Drive API → Habilitar**.
   La API key de Gemini **no** sirve para Drive: Drive necesita una identidad (OAuth).
2. **Google Auth Platform → Branding:** nombre `SIGEM` y correo de asistencia. **Público (Audience):** tipo **Externo**.
   - **Modo Pruebas:** agrega como *usuario de prueba* la cuenta que autorizará. No exige URLs, pero **el token caduca a los 7 días**.
   - **En producción** (*Publicar aplicación*): el token no caduca, pero Google exige URL de página principal y de política de privacidad.
3. **Clientes → Crear cliente → Aplicación de escritorio.** Descargar el JSON a `backend/secrets/oauth-client.json` (carpeta ignorada por Git y por Docker).
4. En `backend/.env`: `ARCHIVOS_BACKEND=drive` y `GOOGLE_DRIVE_FOLDER_ID=<ID o URL de la carpeta>`.
5. `cd backend && uv run python scripts/autorizar_drive.py` → se abre el navegador; iniciar sesión con la cuenta que tiene la carpeta y aceptar la advertencia de app no verificada
   (*Avanzado → Ir a …*). Queda `backend/secrets/oauth-token.json`. El alcance es `drive` completo (`drive.file` no sirve: solo ve archivos creados por la propia app).
6. Enlazar los documentos ya indexados con su PDF:
   ```bash
   DATABASE_URL=postgresql://sigem:sigem@localhost:55432/sigem CHROMA_MODO=http uv run python scripts/sincronizar_drive.py            # simulación
   DATABASE_URL=... CHROMA_MODO=http uv run python scripts/sincronizar_drive.py --aplicar [--subir-faltantes ./carpeta_con_pdfs]
   ```
   El reporte lista tres grupos: enlazados, **en Chroma pero no en Drive** (no se podrán descargar hasta subirlos) y en Drive pero no indexados. Los dudosos no se enlazan.
7. Renovar el token (si caducó): repetir el paso 5.

### Ingesta administrativa y originales
Al indexar un lote, por cada documento: (1) se sube el original (si falla, no se indexa y el admin ve el motivo), (2) embeddings y `upsert` en Chroma, (3) se registra la fila.
Si falla el 2 o el 3 se limpia Chroma y el original va a la papelera. Los omitidos (mismo contenido o título) no se suben. El manifiesto guarda el `id_externo` por archivo, y si el título
no se lee como tipo y número (p. ej. "Decreto 045 de 2026") el resultado incluye una `advertencias` (se encuentra por contenido, no por referencia). Revertir un lote borra de Chroma,
manda los originales a la papelera y elimina las filas; si falla a medias se puede repetir.

---

## [AGREGA] Cómo decide el agente que se pide el archivo: `solicita_archivo` y `preparar_descarga`

El analista añade dos campos a `AnalisisConsulta` (no es una intención nueva: sigue siendo `consulta_sigem` o `seguimiento`, porque igual hay que identificar el documento):

| Mensaje | `solicita_archivo` | `incluye_contenido` | Resultado |
|---|---|---|---|
| "¿Qué dice el Decreto 001 de 2019?" | false | — | Explicación con citas. |
| "Descárgame el Decreto 001 de 2019" / "mándame el PDF" / "¿dónde puedo ver la Ley 387 completa?" | true | false | Solo el botón, con 1–2 frases. |
| "Dame / necesito / quiero / pásame el Decreto 083 de 2018" | true | true | Explicación con citas **y** botón (quien dice "dame el decreto" muchas veces quiere entenderlo). |
| "Pásame el decreto de taxis y dime cuánto cuesta la carrera mínima" | true | true | Ídem. |
| Con documento activo: "pásamelo" / "descárgamelo en PDF" | true | true / false | Entrega el documento activo. |
| "Pásame el archivo" sin referencia ni documento activo | — | — | Pregunta de aclaración. |

**Dos capas de control:** (1) el analista decide; (2) una compuerta determinística: `preparar_descarga` **solo** se agrega a las tools del investigador cuando `solicita_archivo` es true
(en los demás turnos ni existe para el modelo), la API devuelve `archivos: []` si el analista no lo marcó, y la URL la construye el backend, nunca el modelo.

`preparar_descarga(archivo)` valida el PDF contra el catálogo y contra `archivos_documento`; devuelve `archivos` (`titulo`, `archivo`, `url_descarga` relativa) o un error "no disponible para descarga".
El redactor tiene dos modos nuevos (`entrega` y `descarga_no_disponible`) y la regla de **nunca escribir URLs**. Si hay varios documentos posibles se pide aclaración y no se entrega ninguno.

Contrato de la API (aditivo): `ChatResponse` y `ChatVozResponse` ganan `archivos: [{titulo, archivo, url_descarga}]` (por defecto `[]`). `frontend/index.html` pinta un botón
"Descargar <título> (PDF)" por archivo (texto y voz; objetivo táctil ≥ 44 px, foco visible, mensaje amable si la descarga da 404/503).

---

## [REEMPLAZA] "Known pending items"

**Quitar** el punto "Chroma corre embebido en el proceso del backend": ya es un servicio independiente (ver arriba).

Puntos que **quedan** relacionados con esta etapa:
- El botón de descarga no reaparece al recargar un hilo viejo (los `archivos` no se guardan en `turnos`); se ve el texto pero no el botón.
- "documento 18399" (búsqueda por código) solo funciona con los documentos originales, cuyo nombre empieza por el código de SIGEM.
- Los documentos cuyo original no esté en Drive responden "no disponible para descarga" hasta que se suban (`sincronizar_drive.py --subir-faltantes`).
- Si la app OAuth de Drive queda en modo *Pruebas*, el token debe renovarse cada 7 días.
- `admin.html` no muestra el campo `advertencias` del resultado de un lote (solo los errores).
