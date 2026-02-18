from typing import Optional, Sequence, Tuple

from sqlalchemy import (
    Index,
    Integer,
    String,
    Text,
    JSON,
    DateTime,
    ForeignKey,
    select,
    delete,
    func,
    and_,
    or_,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from indexer.domain.models import (
    Chunk,
    RepoRef,
    Anchor,
    Language,
    ChunkKind,
    Visibility,
    SymbolRef,
)


class Base(DeclarativeBase):
    pass


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)

    repo_id: Mapped[str] = mapped_column(String(200), index=True)
    ref: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    commit: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    path: Mapped[str] = mapped_column(Text, index=True)
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)

    language: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)

    symbol_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, index=True)

    text: Mapped[str] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(String(128), index=True)

    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    visibility: Mapped[str] = mapped_column(String(32), index=True)

    defines: Mapped[list[dict]] = mapped_column(JSON, default=list)
    references: Mapped[list[dict]] = mapped_column(JSON, default=list)
    imports: Mapped[list[str]] = mapped_column(JSON, default=list)
    links: Mapped[dict] = mapped_column(JSON, default=dict)


Index("ix_chunks_repo_path", ChunkRow.repo_id, ChunkRow.path)
Index("ix_chunks_repo_ref_path", ChunkRow.repo_id, ChunkRow.ref, ChunkRow.path)
Index("ix_chunks_repo_commit_path", ChunkRow.repo_id, ChunkRow.commit, ChunkRow.path)
Index("ix_chunks_repo_symbol", ChunkRow.repo_id, ChunkRow.symbol_name)


class ChunkEmbeddingStateRow(Base):
    __tablename__ = "chunk_embedding_state"

    chunk_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding_model: Mapped[str] = mapped_column(String(200), primary_key=True)
    text_hash: Mapped[str] = mapped_column(String(128), index=True)
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


Index("ix_embed_state_model", ChunkEmbeddingStateRow.embedding_model)
Index("ix_embed_state_chunk", ChunkEmbeddingStateRow.chunk_id)


class RepoFileStateRow(Base):
    __tablename__ = "repo_file_state"

    repo_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    ref: Mapped[str] = mapped_column(String(200), primary_key=True)
    path: Mapped[str] = mapped_column(Text, primary_key=True)

    content_hash: Mapped[str] = mapped_column(String(128), index=True)
    commit: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


Index("ix_file_state_repo_ref", RepoFileStateRow.repo_id, RepoFileStateRow.ref)
Index("ix_file_state_repo_ref_path", RepoFileStateRow.repo_id, RepoFileStateRow.ref, RepoFileStateRow.path)


def _row_to_chunk(r: ChunkRow) -> Chunk:
    return Chunk(
        chunk_id=r.chunk_id,
        repo=RepoRef(repo_id=r.repo_id, ref=r.ref, commit=r.commit),
        path=r.path,
        language=Language(r.language),
        anchor=Anchor(start_line=r.start_line, end_line=r.end_line),
        kind=ChunkKind(r.kind),
        symbol=None,
        text=r.text,
        text_hash=r.text_hash,
        defines=[SymbolRef.model_validate(x) for x in (r.defines or [])],
        references=[SymbolRef.model_validate(x) for x in (r.references or [])],
        imports=list(r.imports or []),
        links=dict(r.links or {}),
        tags=list(r.tags or []),
        visibility=Visibility(r.visibility),
    )


class MetadataStore:
    def __init__(self, engine: AsyncEngine):
        self._engine = engine
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            expire_on_commit=False,
            autoflush=False,
        )

    async def init_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_file_hashes(self, *, repo_id: str, ref: str) -> dict[str, str]:
        stmt = select(RepoFileStateRow.path, RepoFileStateRow.content_hash).where(
            RepoFileStateRow.repo_id == repo_id,
            RepoFileStateRow.ref == ref,
        )

        async with self._sessionmaker() as s:
            res = await s.execute(stmt)
            rows = res.all()

        return {p: h for (p, h) in rows}

    async def upsert_file_states(
        self,
        *,
        repo_id: str,
        ref: str,
        commit: Optional[str],
        items: Sequence[Tuple[str, str]],
    ) -> int:
        if not items:
            return 0

        values = [
            dict(
                repo_id=repo_id,
                ref=ref,
                path=path,
                content_hash=content_hash,
                commit=commit,
            )
            for (path, content_hash) in items
        ]

        stmt = pg_insert(RepoFileStateRow).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[RepoFileStateRow.repo_id, RepoFileStateRow.ref, RepoFileStateRow.path],
            set_={
                "content_hash": stmt.excluded.content_hash,
                "commit": stmt.excluded.commit,
                "updated_at": func.now(),
            },
        )

        async with self._sessionmaker() as s:
            await s.execute(stmt)
            await s.commit()

        return len(items)

    async def delete_file_states(self, *, repo_id: str, ref: str, paths: Sequence[str]) -> int:
        if not paths:
            return 0

        stmt = (
            delete(RepoFileStateRow)
            .where(
                RepoFileStateRow.repo_id == repo_id,
                RepoFileStateRow.ref == ref,
                RepoFileStateRow.path.in_(list(paths)),
            )
            .returning(RepoFileStateRow.path)
        )

        async with self._sessionmaker() as s:
            res = await s.execute(stmt)
            rows = res.all()
            await s.commit()

        return len(rows)

    async def list_chunk_ids_by_path(self, *, repo_id: str, ref: Optional[str], path: str, commit: Optional[str]) -> list[str]:
        stmt = select(ChunkRow.chunk_id).where(
            ChunkRow.repo_id == repo_id,
            ChunkRow.path == path,
        )
        if ref is not None:
            stmt = stmt.where(ChunkRow.ref == ref)
        if commit is not None:
            stmt = stmt.where(ChunkRow.commit == commit)

        async with self._sessionmaker() as s:
            res = await s.execute(stmt)
            rows = res.all()

        return [cid for (cid,) in rows]

    async def delete_chunks_by_path_return_ids(
        self,
        *,
        repo_id: str,
        ref: Optional[str],
        path: str,
        commit: Optional[str],
    ) -> list[str]:
        stmt = delete(ChunkRow).where(
            ChunkRow.repo_id == repo_id,
            ChunkRow.path == path,
        )
        if ref is not None:
            stmt = stmt.where(ChunkRow.ref == ref)
        if commit is not None:
            stmt = stmt.where(ChunkRow.commit == commit)

        stmt = stmt.returning(ChunkRow.chunk_id)

        async with self._sessionmaker() as s:
            res = await s.execute(stmt)
            rows = res.all()
            await s.commit()

        return [cid for (cid,) in rows]

    async def upsert_chunks(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0

        values = []
        for ch in chunks:
            values.append(
                dict(
                    chunk_id=ch.chunk_id,
                    repo_id=ch.repo.repo_id,
                    ref=ch.repo.ref,
                    commit=ch.repo.commit,
                    path=ch.path,
                    start_line=ch.anchor.start_line,
                    end_line=ch.anchor.end_line,
                    language=ch.language.value,
                    kind=ch.kind.value,
                    symbol_name=(ch.symbol.name if ch.symbol else None),
                    text=ch.text,
                    text_hash=ch.text_hash,
                    tags=list(ch.tags),
                    visibility=ch.visibility.value,
                    defines=[d.model_dump() for d in (ch.defines or [])],
                    references=[r.model_dump() for r in (ch.references or [])],
                    imports=list(ch.imports or []),
                    links=dict(ch.links or {}),
                )
            )

        stmt = pg_insert(ChunkRow).values(values)

        update_cols = {
            c.name: stmt.excluded[c.name]
            for c in ChunkRow.__table__.columns
            if c.name != "chunk_id"
        }

        stmt = stmt.on_conflict_do_update(
            index_elements=[ChunkRow.chunk_id],
            set_=update_cols,
        )

        async with self._sessionmaker() as s:
            await s.execute(stmt)
            await s.commit()

        return len(chunks)

    async def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []

        stmt = select(ChunkRow).where(ChunkRow.chunk_id.in_(list(chunk_ids)))

        async with self._sessionmaker() as s:
            res = await s.execute(stmt)
            rows = res.scalars().all()

        by_id = {r.chunk_id: r for r in rows}
        ordered = [by_id[cid] for cid in chunk_ids if cid in by_id]
        return [_row_to_chunk(r) for r in ordered]

    async def get_chunks_pending_embedding(
        self,
        *,
        repo_id: str,
        embedding_model: str,
        limit: int = 500,
        offset: int = 0,
        ref: Optional[str] = None,
    ) -> list[Chunk]:
        st = ChunkEmbeddingStateRow

        stmt = (
            select(ChunkRow)
            .outerjoin(
                st,
                and_(st.chunk_id == ChunkRow.chunk_id, st.embedding_model == embedding_model),
            )
            .where(ChunkRow.repo_id == repo_id)
        )

        if ref is not None:
            stmt = stmt.where(ChunkRow.ref == ref)

        stmt = (
            stmt.where(or_(st.chunk_id.is_(None), st.text_hash != ChunkRow.text_hash))
            .order_by(ChunkRow.path.asc(), ChunkRow.start_line.asc())
            .limit(limit)
            .offset(offset)
        )

        async with self._sessionmaker() as s:
            res = await s.execute(stmt)
            rows = res.scalars().all()

        return [_row_to_chunk(r) for r in rows]

    async def mark_chunks_embedded(self, *, items: Sequence[Tuple[str, str, str]]) -> int:
        if not items:
            return 0

        values = [
            dict(chunk_id=chunk_id, embedding_model=embedding_model, text_hash=text_hash)
            for (chunk_id, embedding_model, text_hash) in items
        ]

        stmt = pg_insert(ChunkEmbeddingStateRow).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[ChunkEmbeddingStateRow.chunk_id, ChunkEmbeddingStateRow.embedding_model],
            set_={"text_hash": stmt.excluded.text_hash, "updated_at": func.now()},
        )

        async with self._sessionmaker() as s:
            await s.execute(stmt)
            await s.commit()

        return len(items)