# Pinned digest for python:3.12-slim-bookworm (OCI index; multi-arch). Retry deploy if registry times out.
FROM python@sha256:58525e1a8dada8e72d6f8a11a0ddff8d981fd888549108db52455d577f927f77

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY data/ data/
COPY metrics/ metrics/
COPY assets/ assets/
COPY grid_data2.xlsx .

RUN mkdir -p cache

ENV DASH_HOST=0.0.0.0
ENV DASH_PORT=8050

EXPOSE 8050

CMD ["python", "app.py"]
