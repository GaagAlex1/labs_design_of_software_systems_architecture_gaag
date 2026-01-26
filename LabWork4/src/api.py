from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field, HttpUrl, field_validator


app = FastAPI(
    title="RAG Repo Assistant API",
    version="1.0.0",
    openapi_url="/openapi.json",
)

REPOS: Dict[UUID, dict] = {}

JOBS: Dict[UUID, dict] = {}

REPO_JOBS: Dict[UUID, set] = {}


IndexMode = Literal["full", "incremental"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class ErrorDetailsItem(BaseModel):
    field: str
    issue: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[List[ErrorDetailsItem]] = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class RepoCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    vcs_url: HttpUrl
    default_ref: str = Field(..., min_length=1, max_length=200)


class RepoCreateResponse(BaseModel):
    repo_uuid: UUID
    status: Literal["created"]


class RepoItem(BaseModel):
    repo_uuid: UUID
    name: str
    default_ref: str


class RepoUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    default_ref: Optional[str] = Field(None, min_length=1, max_length=200)

    @field_validator("name", "default_ref")
    @classmethod
    def not_blank(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class RepoUpdateResponse(BaseModel):
    repo_uuid: UUID
    status: Literal["updated"]
    name: str
    default_ref: str


class IndexJobCreateRequest(BaseModel):
    ref: str = Field(..., min_length=1, max_length=200)
    mode: IndexMode = "incremental"


class IndexJobCreateResponse(BaseModel):
    job_uuid: UUID
    status: Literal["queued"]


class IndexJobStatusResponse(BaseModel):
    job_uuid: UUID
    status: JobStatus
    progress: float = Field(..., ge=0.0, le=1.0)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


class SearchRequest(BaseModel):
    repo_uuid: UUID
    ref: str = Field(..., min_length=1, max_length=200)
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(5, ge=1, le=50)


class SearchItem(BaseModel):
    chunk_uuid: UUID
    path: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    items: List[SearchItem]


class ChatRequest(BaseModel):
    repo_uuid: UUID
    ref: str = Field(..., min_length=1, max_length=200)
    question: str = Field(..., min_length=1, max_length=5000)
    max_sources: int = Field(3, ge=1, le=20)


class ChatSource(BaseModel):
    path: str
    chunk_uuid: UUID


class ChatResponse(BaseModel):
    answer: str
    sources: List[ChatSource]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def raise_404(message: str = "Not found") -> None:
    raise HTTPException(status_code=404, detail=message)


def raise_400(message: str = "Bad request") -> None:
    raise HTTPException(status_code=400, detail=message)


def raise_409(message: str = "Conflict") -> None:
    raise HTTPException(status_code=409, detail=message)


def get_repo_or_404(repo_id: UUID) -> dict:
    repo = REPOS.get(repo_id)
    if not repo:
        raise_404("Repository not found")
    return repo


def any_active_jobs_for_repo(repo_id: UUID) -> bool:
    job_ids = REPO_JOBS.get(repo_id, set())
    for jid in job_ids:
        job = JOBS.get(jid)
        if job and job["status"] in ("queued", "running"):
            return True
    return False


def simulate_job_progress(job: dict) -> dict:
    status = job["status"]
    progress = job["progress"]

    if status == "queued":
        job["status"] = "running"
        job["started_at"] = job["started_at"] or now_utc()
        job["progress"] = max(progress, 0.05)
        return job

    if status == "running":
        new_progress = min(1.0, progress + 0.2)
        job["progress"] = new_progress
        if new_progress >= 1.0:
            job["status"] = "succeeded"
            job["finished_at"] = now_utc()
        return job

    return job


def build_validation_error(field: str, issue: str) -> HTTPException:
    return HTTPException(status_code=400, detail={"field": field, "issue": issue})


@app.post("/api/v1/repos", response_model=RepoCreateResponse, status_code=201)
def create_repo(payload: RepoCreateRequest):
    # Conflict rule: same vcs_url already exists
    for repo in REPOS.values():
        if str(repo["vcs_url"]) == str(payload.vcs_url):
            raise_409("Repository with same vcs_url already registered")

    repo_id = uuid4()
    REPOS[repo_id] = {
        "repo_uuid": repo_id,
        "name": payload.name,
        "vcs_url": str(payload.vcs_url),
        "default_ref": payload.default_ref,
        "created_at": now_utc(),
        "updated_at": now_utc(),
    }
    REPO_JOBS.setdefault(repo_id, set())
    return RepoCreateResponse(repo_uuid=repo_id, status="created")


@app.get("/api/v1/repos", response_model=List[RepoItem])
def list_repos():
    return [
        RepoItem(
            repo_uuid=repo["repo_uuid"],
            name=repo["name"],
            default_ref=repo["default_ref"],
        )
        for repo in REPOS.values()
    ]


@app.post(
    "/api/v1/repos/{repo_id}/index-jobs",
    response_model=IndexJobCreateResponse,
    status_code=202,
)
def create_index_job(
    repo_id: UUID = Path(..., description="Repository UUID"),
    payload: IndexJobCreateRequest = ...,
):
    _ = get_repo_or_404(repo_id)

    if any_active_jobs_for_repo(repo_id):
        raise_409("Index job already running for this repository")

    job_id = uuid4()
    JOBS[job_id] = {
        "job_uuid": job_id,
        "repo_uuid": repo_id,
        "ref": payload.ref,
        "mode": payload.mode,
        "status": "queued",
        "progress": 0.0,
        "started_at": None,
        "finished_at": None,
        "error": None,
        "created_at": now_utc(),
    }
    REPO_JOBS.setdefault(repo_id, set()).add(job_id)

    return IndexJobCreateResponse(job_uuid=job_id, status="queued")


@app.get("/api/v1/index-jobs/{job_id}", response_model=IndexJobStatusResponse)
def get_index_job_status(job_id: UUID = Path(..., description="Job UUID")):
    job = JOBS.get(job_id)
    if not job:
        raise_404("Index job not found")

    job = simulate_job_progress(job)

    return IndexJobStatusResponse(
        job_uuid=job["job_uuid"],
        status=job["status"],
        progress=job["progress"],
        started_at=job["started_at"],
        finished_at=job["finished_at"],
        error=job["error"],
    )


@app.post("/api/v1/search", response_model=SearchResponse)
def search(payload: SearchRequest):
    _ = get_repo_or_404(payload.repo_uuid)

    items: List[SearchItem] = []
    for i in range(payload.top_k):
        items.append(
            SearchItem(
                chunk_uuid=uuid4(),
                path="analytics/src/chunk_analyzer.py" if i == 0 else f"docs/part_{i}.md",
                score=round(0.82 - i * 0.03, 3),
                snippet="..." if i else "def analyze_chunks(...):\n    ...",
            )
        )
    return SearchResponse(items=items)


@app.post("/api/v1/chat/completions", response_model=ChatResponse)
def chat(payload: ChatRequest):
    _ = get_repo_or_404(payload.repo_uuid)

    sources: List[ChatSource] = []
    for _ in range(payload.max_sources):
        sources.append(
            ChatSource(
                path=".gitlab-ci.yml",
                chunk_uuid=uuid4(),
            )
        )

    answer = (
        "В реальной реализации здесь формируется RAG-контекст "
        "по репозиторию и вызывается LLM. Ответ должен ссылаться на найденные источники."
    )

    return ChatResponse(answer=answer, sources=sources)


@app.put("/api/v1/repos/{repo_id}", response_model=RepoUpdateResponse)
def update_repo(
    repo_id: UUID = Path(..., description="Repository UUID"),
    payload: RepoUpdateRequest = ...,
):
    repo = get_repo_or_404(repo_id)

    if any_active_jobs_for_repo(repo_id):
        raise_409("Repository settings cannot be updated while index job is running")

    # If name is being changed: ensure uniqueness (optional rule)
    if payload.name is not None:
        for rid, r in REPOS.items():
            if rid != repo_id and r["name"] == payload.name:
                raise_409("Repository name already in use")

    if payload.name is not None:
        repo["name"] = payload.name
    if payload.default_ref is not None:
        repo["default_ref"] = payload.default_ref

    repo["updated_at"] = now_utc()

    return RepoUpdateResponse(
        repo_uuid=repo_id,
        status="updated",
        name=repo["name"],
        default_ref=repo["default_ref"],
    )


@app.delete("/api/v1/repos/{repo_id}", status_code=204)
def delete_repo(repo_id: UUID = Path(..., description="Repository UUID")):
    repo = REPOS.get(repo_id)
    if not repo:
        raise_404("Repository not found")

    if any_active_jobs_for_repo(repo_id):
        raise_409("Repository cannot be deleted while index job is running")

    for jid in list(REPO_JOBS.get(repo_id, set())):
        JOBS.pop(jid, None)
    REPO_JOBS.pop(repo_id, None)
    REPOS.pop(repo_id, None)

    return
