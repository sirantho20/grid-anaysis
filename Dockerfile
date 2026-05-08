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
