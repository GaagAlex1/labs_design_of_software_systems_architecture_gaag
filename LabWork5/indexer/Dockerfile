FROM python:3.11-slim

WORKDIR /app

COPY indexer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY indexer/ ./indexer/

CMD ["python", "-m", "indexer", "run"]