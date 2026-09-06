# Models and the index get baked in at build time, so the running container
# needs no network. Also means a demo can't die on a model download timing out.

FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --only-binary=:all: -r requirements.txt

# sentence-transformers pulls torch, and torch drags the CUDA runtime along with
# it: ~2 GB of wheels to run a 33M-parameter model on CPU. fastembed runs the
# same weights through ONNX. Check for torch instead of assuming it's absent; a
# transitive dep can pull it back in and the only symptom is a fat image.
RUN if pip show torch >/dev/null 2>&1; then echo "torch was pulled in; see note above" >&2; exit 1; fi


FROM python:3.13-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/opt/models

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY app ./app
COPY scripts ./scripts
COPY data ./data

# Download the embedding and reranking models into the image.
RUN python -c "from app.embed import get_model; get_model()" && \
    python -c "from app.rerank import get_reranker; get_reranker()"

# Build the index at image build time.
RUN python scripts/ingest.py

RUN useradd --create-home --uid 1000 app && chown -R app:app /srv /opt/models
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import httpx;httpx.get('http://localhost:8501/_stcore/health',timeout=4).raise_for_status()"

CMD ["streamlit", "run", "app/ui.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
