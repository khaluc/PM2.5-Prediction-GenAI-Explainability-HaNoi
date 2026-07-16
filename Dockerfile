FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app

WORKDIR /app

COPY requirements-runtime.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-runtime.txt

COPY --chown=app:app . .
RUN mkdir -p artifacts/logs artifacts/alerts artifacts/reports output/pdf \
    && chown -R app:app artifacts output data

USER app

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
