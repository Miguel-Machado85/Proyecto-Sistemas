FROM ghcr.io/astral-sh/uv:0.11.23 AS uv
FROM python:3.14-slim

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY agente_sigem ./agente_sigem
COPY main.py ./
RUN uv sync --locked --no-dev

RUN chown -R app:app /app
USER app

EXPOSE 8001
CMD ["uvicorn", "agente_sigem.api:app", "--host", "0.0.0.0", "--port", "8001"]