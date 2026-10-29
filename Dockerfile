FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    SOURCED_INDEX=/app/index/index.db

WORKDIR /app
RUN useradd --system --uid 1001 app

COPY pyproject.toml README.md requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src src
RUN pip install --no-cache-dir --no-deps . && pip check

# Bake the models and the sample index into the image so the container starts
# answering immediately, with no network access needed at runtime.
COPY data data
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')" \
 && sourced ingest data/handbook \
 && chown -R app /app

USER app
EXPOSE 8000
CMD ["uvicorn", "sourced.api:app", "--host", "0.0.0.0", "--port", "8000"]
