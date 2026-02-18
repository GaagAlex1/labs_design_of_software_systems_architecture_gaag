FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir pytest requests asyncpg pymilvus

COPY itests /app/itests
