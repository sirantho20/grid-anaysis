FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DASH_HOST=0.0.0.0 \
    DASH_PORT=8050

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY --chown=app:app ["app.py", "grid_data2.xlsx", "/app/"]
COPY --chown=app:app data/ /app/data/
COPY --chown=app:app metrics/ /app/metrics/
COPY --chown=app:app assets/ /app/assets/

RUN mkdir -p /app/cache && chown -R app:app /app

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; port = os.environ.get('DASH_PORT', '8050'); urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=3)"

USER app

# Force the Dash server to bind to all interfaces inside the container so
# published ports work from every host address.
CMD ["python", "app.py"]
