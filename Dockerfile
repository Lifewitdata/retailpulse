FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY sql/ sql/
COPY src/ src/

# The raw event data and the DuckDB warehouse are generated at container start
# (they are gitignored artifacts), then the API is served. The whole pipeline runs
# offline and deterministically from seed 42 with no LLM key present.
EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health')" || exit 1

CMD ["sh", "-c", "python -m src.analyze --no-llm && uvicorn src.api:app --host 0.0.0.0 --port 8010"]
