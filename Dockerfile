FROM python:3.13-slim AS dependencies

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VINNILAL_DB_PATH=/app/data/telemetry.db

WORKDIR /app
COPY --from=dependencies /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser analytics ./analytics
COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser digital_twin ./digital_twin
COPY --chown=appuser:appuser simulator ./simulator
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser frontend ./frontend

USER appuser
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
