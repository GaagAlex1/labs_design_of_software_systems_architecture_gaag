# Лабораторная работа 3  
## Использование принципов проектирования на уровне методов и классов

## Цель работы
Получить опыт проектирования и реализации модулей с использованием принципов **KISS**, **YAGNI**, **DRY**, **SOLID** и дополнительных принципов (**BDUF**, **SoC**, **MVP**, **PoC**) на примере LLM+RAG ассистента для работы с кодовой базой.

---

## 1. Архитектурные диаграммы

### 1.1. Диаграмма контейнеров
![Диаграмма контейнеров LLM+RAG ассистента](../LabWork2/containers-diagram.png)

### 1.2. Диаграмма компонентов
![Диаграмма компонентов RAG-сервиса](../LabWork2/components-rag-service.png)

---

## 2. Диаграмма последовательностей

**Вариант использования:** «Разработчик задаёт вопрос по коду и получает ответ от ассистента»

![Диаграмма последовательностей](seq-diagram.png)

---

## 3. Диаграмма классов
![Диаграмма классов](class-diagram.png)

---

## 4. Применение основных принципов разработки

### 4.1. KISS (Keep It Simple, Stupid)

**Суть принципа в рамках работы:** каждый модуль делает минимум необходимого и не содержит скрытой логики, не относящейся к его ответственности.

**Где применён:**
- `server/api.py` — HTTP-эндпоинт выполняет только маппинг DTO → доменная модель и делегирует выполнение в `RagPipeline`.
- `server/pipeline.py` — `QueryPreprocessor` выполняет только нормализацию текста запроса.

**Почему это KISS:**
- Точка входа (API) не «знает» деталей RAG (поиск/переранк/промпт/LLM), поэтому остаётся простой и стабильной.
- Конвейер состоит из небольших шагов, каждый шаг изолирован и тестируем.

```python
# server/api.py
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

from .pipeline import RagPipeline, Query, Answer

app = FastAPI()

class SearchRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)

class SearchResponse(BaseModel):
    answer: str

def get_pipeline() -> RagPipeline:
    # KISS: сборка зависимостей вынесена в infrastructure-слой
    from .infrastructure import build_pipeline
    return build_pipeline()

@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest, pipeline: RagPipeline = Depends(get_pipeline)):
    domain_query = Query(text=req.query, user_id=req.user_id)
    answer: Answer = await pipeline.answer(domain_query)
    return SearchResponse(answer=answer.text)
```

---

### 4.2. YAGNI (You Aren’t Gonna Need It)

**Суть принципа в рамках работы:** не добавлять функциональность «на будущее», пока это не требуется сценарием лабораторной работы.

**Где применён:**
- `server/pipeline.py` — реализуется только необходимая предобработка запроса (нормализация пробелов), без «умной» диалоговой памяти и сложных политик.
- `server/infrastructure.py` — сборка ограничена компонентами, которые реально используются в сценарии «одиночный вопрос → ответ».

**Что сознательно НЕ реализовано (обоснованный отказ):**
- Сложная система динамических политик доступа на уровне отдельных фрагментов кода.
- Универсальные плагины под произвольные типы документов (в рамках ЛР нужен код и базовая документация).
- Отдельный модуль диалоговой памяти (контекст — одиночный вопрос).

```python
# server/pipeline.py
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Query:
    text: str
    user_id: str

@dataclass(frozen=True, slots=True)
class Answer:
    text: str

class QueryPreprocessor:
    """YAGNI: минимальная предобработка для учебного сценария."""
    def preprocess(self, query: Query) -> Query:
        normalized = " ".join(query.text.split())
        return Query(text=normalized, user_id=query.user_id)
```

---

### 4.3. DRY (Don’t Repeat Yourself)

**Суть принципа в рамках работы:** общая логика обработки запроса не дублируется между слоями и точками входа, а сосредоточена в едином конвейере.

**Где применён:**
- `server/pipeline.py` — единый `RagPipeline.answer()` содержит общий алгоритм обработки: preprocess → retrieve → rerank → build_context → build_prompt → generate.
- `server/api.py` — эндпоинт не содержит копии логики RAG и не повторяет построение промпта/контекста.

**Почему это DRY:**
- При добавлении новых API-методов (например, `/api/search_debug`) алгоритм переиспользуется через `RagPipeline`, без копирования шагов.
- Формирование промпта инкапсулировано в одном месте (`_build_prompt`).

```python
# server/pipeline.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    text: str
    score: float
    meta: dict[str, Any] | None = None

class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, query: Query) -> list[Chunk]:
        ...

class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: Query, chunks: list[Chunk]) -> list[Chunk]:
        ...

class ContextBuilder(ABC):
    @abstractmethod
    def build(self, chunks: Iterable[Chunk]) -> str:
        ...

class LlmClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        ...

class RagPipeline:
    """
    DRY: единый конвейер обработки запроса.
    SOLID: конвейер только оркестрирует шаги, не содержит инфраструктурных деталей.
    """
    def __init__(
        self,
        preprocessor: QueryPreprocessor,
        retriever: Retriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        llm_client: LlmClient,
    ) -> None:
        self._pre = preprocessor
        self._retriever = retriever
        self._reranker = reranker
        self._context_builder = context_builder
        self._llm = llm_client

    async def answer(self, query: Query) -> Answer:
        q = self._pre.preprocess(query)
        chunks = await self._retriever.retrieve(q)
        ranked = await self._reranker.rerank(q, chunks)
        context = self._context_builder.build(ranked)
        prompt = self._build_prompt(q, context)
        raw = await self._llm.generate(prompt)
        return Answer(text=self._postprocess(raw))

    def _build_prompt(self, query: Query, context: str) -> str:
        return (
            "You are a codebase assistant.\n"
            "Answer the user's question using ONLY the relevant code context.\n\n"
            f"User question:\n{query.text}\n\n"
            f"Relevant code context:\n{context}\n"
        )

    def _postprocess(self, raw: str) -> str:
        return raw.strip()
```

---

### 4.4. SOLID

Ниже показано, как принципы SOLID реализуются на уровне классов и взаимодействий.

#### S — Single Responsibility Principle (SRP)

**Где применён:**
- `QueryPreprocessor` — только нормализация запроса.
- `RagPipeline` — только оркестрация шагов (без деталей хранения/сети).
- `VectorStoreRetriever` — только получение кандидатов из векторного хранилища.

```python
# server/retriever.py
from .pipeline import Query, Chunk, Retriever

class VectorStoreRetriever(Retriever):
    """SRP: отвечает только за retrieval из векторного хранилища."""
    def __init__(self, vector_store_client, top_k: int = 30) -> None:
        self._store = vector_store_client
        self._top_k = top_k

    async def retrieve(self, query: Query) -> list[Chunk]:
        rows = await self._store.search(query.text, top_k=self._top_k)
        return [
            Chunk(
                id=str(r["id"]),
                text=str(r["text"]),
                score=float(r.get("score", 0.0)),
                meta=r.get("meta"),
            )
            for r in rows
        ]
```

#### O — Open/Closed Principle (OCP)

**Где применён:**
- Расширение поведения достигается добавлением новых реализаций интерфейсов `Retriever/Reranker/ContextBuilder/LlmClient` без изменения кода `RagPipeline`.

**Пример расширения без модификации `RagPipeline`:**
- можно добавить `Bm25Retriever(Retriever)` или `HybridRetriever(Retriever)` и подключить через `build_pipeline()`.

#### L — Liskov Substitution Principle (LSP)

**Где применён:**
- `RagPipeline` работает с абстракциями (`Retriever`, `Reranker`, `LlmClient`) и ожидает соблюдения контрактов.
- Реализации не добавляют более строгие предусловия, чем объявлено в интерфейсе (например, не требуют «особый формат query.text»).

#### I — Interface Segregation Principle (ISP)

**Где применён:**
- Интерфейсы разделены по ролям: retrieval / rerank / context / llm.
- Компоненты зависят только от того, что им действительно нужно (например, `ContextBuilder` не обязан иметь методы поиска или генерации).

#### D — Dependency Inversion Principle (DIP)

**Где применён:**
- Высокоуровневый модуль `RagPipeline` зависит от абстракций.
- Конкретные зависимости создаются в инфраструктурном слое (`build_pipeline`), а затем внедряются в `RagPipeline`.

```python
# server/infrastructure.py
from .pipeline import RagPipeline, QueryPreprocessor
from .retriever import VectorStoreRetriever

class SimpleReranker:
    async def rerank(self, query, chunks):
        # YAGNI: простая сортировка по score для учебной версии
        return sorted(chunks, key=lambda c: c.score, reverse=True)

class SimpleContextBuilder:
    def __init__(self, max_chars: int = 6000) -> None:
        self._max_chars = max_chars

    def build(self, chunks) -> str:
        buf = []
        size = 0
        for c in chunks:
            part = f"\n---\n# chunk={c.id} score={c.score}\n{c.text}\n"
            if size + len(part) > self._max_chars:
                break
            buf.append(part)
            size += len(part)
        return "".join(buf)

class DummyLlmClient:
    async def generate(self, prompt: str) -> str:
        # В реальной системе здесь интеграция с провайдером LLM.
        return "Stub answer (replace with real LLM integration)."

def build_pipeline() -> RagPipeline:
    pre = QueryPreprocessor()

    # vector_store_client должен предоставлять async search(query: str, top_k: int)
    vector_store_client = ...  # инициализируется здесь
    retriever = VectorStoreRetriever(vector_store_client, top_k=30)

    reranker = SimpleReranker()
    context_builder = SimpleContextBuilder(max_chars=6000)
    llm = DummyLlmClient()

    return RagPipeline(
        preprocessor=pre,
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        llm_client=llm,
    )
```

---

## 5. Дополнительные принципы разработки

### 5.1. BDUF (Big Design Up Front — «Масштабное проектирование прежде всего»)

**Кратко:** попытка заранее детально спроектировать всю систему до реализации.

**Применимость / решение для ЛР: частично применён.**
- Полный BDUF для учебного проекта не используется: это повышает стоимость изменений и приводит к избыточной сложности.
- Использован ограниченный BDUF на уровне: контейнерной архитектуры, компонентной декомпозиции, базовых контрактов между компонентами (`Retriever/Reranker/LlmClient`).

**Почему так:**
- Этого достаточно, чтобы избежать архитектурных ошибок (жёстких связей, невозможности замены компонентов).
- Детали (точная структура индекса, оптимальные промпты, сложные политики доступа) целесообразно уточнять итеративно по мере тестирования.

---

### 5.2. SoC (Separation of Concerns — «Разделение ответственности»)

**Кратко:** разные аспекты системы (UI, бизнес-логика, инфраструктура) должны быть разделены, чтобы изменения в одном не ломали другие.

**Где применён:**
- `server/api.py` — слой доставки (HTTP): только I/O и маппинг DTO.
- `server/pipeline.py` — прикладной слой (use-case): оркестрация RAG-шага.
- `server/retriever.py`, `server/infrastructure.py` — инфраструктура: доступ к векторному хранилищу и сборка зависимостей.

**Польза:**
- Меняется внешний протокол (например, REST → gRPC) без переписывания алгоритма RAG.
- Меняется векторная БД/ретривер без затрагивания API и use-case.

---

### 5.3. MVP (Minimum Viable Product — «Минимально жизнеспособный продукт»)

**Кратко:** минимальный набор функциональности, который уже даёт ценность пользователю и позволяет быстро собрать обратную связь.

**Как проявляется в работе:**
- Реализован базовый сценарий: пользователь задаёт вопрос → retrieval → rerank → ответ LLM.
- Минимальная предобработка запроса и постобработка ответа.
- Минимальная инфраструктура: единая точка входа API и базовая сборка пайплайна.

**Почему это MVP:**
- Быстро проверяется полезность ассистента на реальных запросах.
- Сохраняются точки расширения (интерфейсы), но не добавляется избыточная функциональность.

---

### 5.4. PoC (Proof of Concept — «Доказательство концепции»)

**Кратко:** проверка технической реализуемости ключевого решения до того, как инвестировать время в полноценный продукт.

**Где PoC целесообразен:**
- качество векторного поиска по реальному монорепозиторию;
- скорость ответа при типичной нагрузке;
- интеграция с выбранным провайдером LLM.

**Форма PoC (варианты):**
- небольшой скрипт, индексирующий один сервис и проверяющий качество поиска;
- прототип без полноценной авторизации и UI: только API и простой CLI;
- изолированный тест LLM-клиента на реальных примерах кода.
