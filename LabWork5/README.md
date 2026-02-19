# Monorepo Assistant (indexer + infra)

Набор контейнеров для индексатора репозитория и интеграционных тестов: Postgres + Milvus (через etcd/minio) + Triton + indexer + itests.

## Какие контейнеры используются

- **postgres** — метаданные/служебное хранилище для indexer  
- **etcd** — зависимость Milvus  
- **minio** — объектное хранилище для Milvus  
- **milvus** — векторная БД (коллекция `chunks`)  
- **triton** — Triton Inference Server с моделью эмбеддингов (например `qwen3_embedder`)  
- **indexer** — индексация репозитория (`./repo`) и запись чанков/эмбеддингов в Milvus  
- **itests** — интеграционные тесты (pytest)

## Требования

- Docker + Docker Compose v2
- NVIDIA GPU + NVIDIA Container Toolkit (если Triton использует GPU)
- Папка `./repo` (репозиторий, который индексируем)

## Запуск

### 1) Поднять только инфраструктуру (без indexer/itests)

```bash
docker compose up -d --build
```

Остановить/удалить контейнеры:

```bash
docker compose down
```

### 2) Прогон интеграционного профиля (indexer + itests)

`indexer` требует переменную `INDEXER_REPO_COMMIT` (SHA коммита репозитория в `./repo`).

```bash
INDEXER_REPO_COMMIT="$(cd repo && git rev-parse HEAD)" \
docker compose --profile integration_testing up --build \
--exit-code-from itests
```

- Команда завершится с кодом тестов `itests`
- После завершения контейнеры остановятся (из‑за `--abort-on-container-exit`)

Очистка после прогона:

```bash
docker compose down
```

### 3) Публикация образа сервиса indexer на Dockerhub

```bash
docker login

docker build -f indexer/Dockerfile -t galex134/indexer:0.1.0 .
docker push galex134/indexer:0.1.0
```

Образ доступен на https://hub.docker.com/repository/docker/galex134/indexer/tags/0.1.0