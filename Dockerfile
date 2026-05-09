# Tag-based base for reliable registry resolution on builders (e.g. CapRover). For strict reproducibility,
# optionally repin with an architecture-specific digest once builds are stable (inspect with:
#   docker buildx imagetools inspect python:3.12-slim-bookworm
# then pin e.g. FROM python@sha256:<digest matching your server's OS/ARCH output from build logs>).
FROM python:3.12-slim-bookworm

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
