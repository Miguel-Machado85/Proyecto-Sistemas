FROM ghcr.io/astral-sh/uv:0.11.23 AS uv
FROM python:3.14-slim

COPY --from=uv /uv /uvx /bin/

# HF_HOME: caché de Hugging Face del reranker. Debe ser escribible por el usuario `app`
# y se monta como volumen (hf-cache) para no volver a descargar el modelo al recrear el contenedor.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app

# Dependencias primero (capa cacheable). torch se instala desde el índice CPU declarado en
# pyproject.toml ([[tool.uv.index]] pytorch-cpu): sin CUDA, la imagen es mucho más liviana.
# uv.lock es opcional (está en .gitignore): si existe se respeta, si no uv resuelve desde pyproject.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

COPY README.md ./
COPY agente_sigem ./agente_sigem
COPY main.py ./
RUN uv sync --no-dev

# Solo la caché del reranker necesita ser escribible por `app`. NO usar `chown -R /app`: copiaría el
# venv completo (~2 GB) en una capa nueva y duplicaría el tamaño de la imagen.
RUN mkdir -p /app/.cache/huggingface && chown -R app:app /app/.cache
USER app

EXPOSE 8001
CMD ["uvicorn", "agente_sigem.api:app", "--host", "0.0.0.0", "--port", "8001"]
