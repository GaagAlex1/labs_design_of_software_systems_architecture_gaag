from typing import Optional

from sqlalchemy.ext.asyncio import create_async_engine
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from indexer.domain.models import RepoRef
from indexer.reader import Reader
from indexer.chunker import Chunker
from indexer.embedder import Embedder, EmbedderConfig
from indexer.storage.metadata_store import MetadataStore
from indexer.storage.vector_store import VectorStore, MilvusConfig


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INDEXER_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="forbid",
    )

    repo_root: str = Field(..., min_length=1)
    repo_id: str = Field(..., min_length=1)
    repo_ref: str = Field(default="main", min_length=1)
    repo_commit: Optional[str] = None

    postgres_dsn: str = Field(..., min_length=1)

    milvus_uri: str = "http://localhost:19530"
    milvus_token: Optional[str] = None
    milvus_collection: str = "chunks"

    embedding_model: str = "default"
    embedding_dim: int = Field(..., ge=8, le=8192)

    triton_url: str = "localhost:8000"
    triton_model_name: str = Field(..., min_length=1)
    triton_input_name: str = "TEXT"
    triton_output_name: str = "EMBEDDING"

    pending_limit: int = Field(default=2000, ge=1, le=1_000_000)


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        self.engine = create_async_engine(cfg.postgres_dsn, pool_pre_ping=True)
        self.meta = MetadataStore(self.engine)

        self.vector_store = VectorStore(
            MilvusConfig(
                uri=cfg.milvus_uri,
                token=cfg.milvus_token,
                collection=cfg.milvus_collection,
                embedding_dim=cfg.embedding_dim,
                store_text=False,
            )
        )

        self.embedder = Embedder(
            EmbedderConfig(
                url=cfg.triton_url,
                model_name=cfg.triton_model_name,
                input_name=cfg.triton_input_name,
                output_name=cfg.triton_output_name,
                embedding_dim=cfg.embedding_dim,
            )
        )

        self.reader = Reader(
            repo_root=cfg.repo_root,
            repo=RepoRef(repo_id=cfg.repo_id, ref=cfg.repo_ref, commit=cfg.repo_commit),
        )
        self.chunker = Chunker()

    async def initialize_db(self) -> None:
        await self.meta.init_schema()

    async def run(self) -> dict:
        repo_id = self.cfg.repo_id
        ref = self.cfg.repo_ref
        commit = self.cfg.repo_commit

        prev = await self.meta.get_file_hashes(repo_id=repo_id, ref=ref)

        files = self.reader.read_repo()
        cur = {sf.manifest.path: (sf.manifest.content_hash or "") for sf in files}

        deleted_paths = [p for p in prev.keys() if p not in cur]
        changed_files = [sf for sf in files if prev.get(sf.manifest.path) != (sf.manifest.content_hash or "")]

        chunks_deleted = 0

        for path in deleted_paths:
            old_ids = await self.meta.delete_chunks_by_path_return_ids(
                repo_id=repo_id,
                ref=ref,
                path=path,
                commit=None,
            )
            if old_ids:
                self.vector_store.delete(old_ids)
                chunks_deleted += len(old_ids)

        if deleted_paths:
            await self.meta.delete_file_states(repo_id=repo_id, ref=ref, paths=deleted_paths)

        chunks_upserted = 0
        changed_paths: list[str] = []

        for sf in changed_files:
            path = sf.manifest.path

            old_ids = await self.meta.delete_chunks_by_path_return_ids(
                repo_id=repo_id,
                ref=ref,
                path=path,
                commit=None,
            )
            if old_ids:
                self.vector_store.delete(old_ids)
                chunks_deleted += len(old_ids)

            new_chunks = self.chunker.chunk(sf)
            chunks_upserted += await self.meta.upsert_chunks(new_chunks)

            changed_paths.append(path)

        if changed_paths:
            await self.meta.upsert_file_states(
                repo_id=repo_id,
                ref=ref,
                commit=commit,
                items=[(p, cur[p]) for p in changed_paths],
            )

        pending = await self.meta.get_chunks_pending_embedding(
            repo_id=repo_id,
            ref=ref,
            embedding_model=self.cfg.embedding_model,
            limit=self.cfg.pending_limit,
            offset=0,
        )

        embedded = 0
        if pending:
            vectors = self.embedder.embed_texts([c.text for c in pending])

            self.vector_store.upsert(pending, vectors, self.cfg.embedding_model)

            await self.meta.mark_chunks_embedded(
                items=[(c.chunk_id, self.cfg.embedding_model, c.text_hash) for c in pending]
            )
            embedded = len(pending)

        return {
            "files_total": len(files),
            "files_changed": len(changed_files),
            "files_deleted": len(deleted_paths),
            "chunks_deleted": chunks_deleted,
            "chunks_upserted": chunks_upserted,
            "chunks_pending": len(pending),
            "chunks_embedded": embedded,
        }

    async def close(self) -> None:
        await self.engine.dispose()