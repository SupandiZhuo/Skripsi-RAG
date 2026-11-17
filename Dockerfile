FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ANONYMIZED_TELEMETRY=False \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.13.8

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project --python /usr/local/bin/python

# Copy only application code and public examples; keep private inputs out of the image.
COPY app.py ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY evaluation/ ./evaluation/
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY data/examples/ ./data/examples/
RUN uv sync --locked --no-dev --no-editable --python /usr/local/bin/python \
    && uv cache clean

RUN useradd --create-home --user-group --uid 10001 app \
    && mkdir -p /app/chroma_db /app/.cache/nvd /home/app/.cache/huggingface \
    && chown -R app:app /app/data /app/chroma_db /app/.cache /home/app/.cache

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
