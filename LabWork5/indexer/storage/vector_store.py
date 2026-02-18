from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from indexer.domain.models import Chunk


class MilvusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str = Field(default="http://localhost:19530", min_length=1, max_length=2048)
    token: Optional[str] = Field(default=None, max_length=512)

    collection: str = Field(default="chunks", min_length=1, max_length=128)
    embedding_dim: int = Field(..., ge=8, le=8192)

    metric_type: str = Field(default="IP")
    hnsw_M: int = Field(default=16, ge=4, le=128)
    hnsw_efConstruction: int = Field(default=200, ge=16, le=2000)
    search_ef: int = Field(default=128, ge=8, le=4096)

    store_text: bool = Field(default=False)
    max_path_len: int = Field(default=1024, ge=64, le=8192)
    max_text_len: int = Field(default=8192, ge=256, le=65535)


class VectorSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    distance: float
    entity: Dict[str, Any] = Field(default_factory=dict)


class VectorStore:
    def __init__(self, cfg: MilvusConfig):
        from pymilvus import MilvusClient

        self.cfg = cfg
        self.client = MilvusClient(uri=cfg.uri, token=cfg.token)

        if not self.client.has_collection(cfg.collection):
            self._create_collection()

        self.client.load_collection(collection_name=cfg.collection)

    def _create_collection(self) -> None:
        from pymilvus import MilvusClient, DataType

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(field_name="repo_id", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=self.cfg.max_path_len)
        schema.add_field(field_name="language", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="kind", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="visibility", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="start_line", datatype=DataType.INT64)
        schema.add_field(field_name="end_line", datatype=DataType.INT64)
        schema.add_field(field_name="text_hash", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="embedding_model", datatype=DataType.VARCHAR, max_length=200)

        if self.cfg.store_text:
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=self.cfg.max_text_len)

        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.cfg.embedding_dim)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            index_name="embedding_hnsw",
            metric_type=self.cfg.metric_type,
            params={"M": self.cfg.hnsw_M, "efConstruction": self.cfg.hnsw_efConstruction},
        )

        self.client.create_collection(
            collection_name=self.cfg.collection,
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]], embedding_model: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks and vectors length mismatch: {len(chunks)} != {len(vectors)}")

        data: List[Dict[str, Any]] = []
        for ch, vec in zip(chunks, vectors):
            if len(vec) != self.cfg.embedding_dim:
                raise ValueError(
                    f"embedding dim mismatch for chunk_id={ch.chunk_id}: {len(vec)} != {self.cfg.embedding_dim}"
                )

            row: Dict[str, Any] = {
                "chunk_id": ch.chunk_id,
                "repo_id": ch.repo.repo_id,
                "path": (ch.path or "")[: self.cfg.max_path_len],
                "language": str(ch.language),
                "kind": str(ch.kind),
                "visibility": str(ch.visibility),
                "start_line": int(ch.anchor.start_line),
                "end_line": int(ch.anchor.end_line),
                "text_hash": (ch.text_hash or "")[:128],
                "embedding_model": embedding_model[:200],
                "embedding": vec,
            }
            if self.cfg.store_text:
                row["text"] = (ch.text or "")[: self.cfg.max_text_len]

            data.append(row)

        self.client.upsert(collection_name=self.cfg.collection, data=data)

    def delete(self, chunk_ids: List[str]) -> None:
        if not chunk_ids:
            return

        ids = ",".join(f'"{cid}"' for cid in chunk_ids)
        self.client.delete(collection_name=self.cfg.collection, filter=f'chunk_id in [{ids}]')

    def query(
        self,
        query_vector: List[float],
        k: int = 10,
        filter_expr: str = "",
        output_fields: Optional[List[str]] = None,
    ) -> List[VectorSearchResult]:
        if len(query_vector) != self.cfg.embedding_dim:
            raise ValueError(f"query dim mismatch: {len(query_vector)} != {self.cfg.embedding_dim}")

        if output_fields is None:
            output_fields = [
                "repo_id",
                "path",
                "language",
                "kind",
                "visibility",
                "start_line",
                "end_line",
                "text_hash",
                "embedding_model",
            ]
            if self.cfg.store_text:
                output_fields.append("text")

        search_params = {
            "metric_type": self.cfg.metric_type,
            "params": {"ef": self.cfg.search_ef},
        }

        res = self.client.search(
            collection_name=self.cfg.collection,
            data=[query_vector],
            anns_field="embedding",
            filter=filter_expr,
            limit=k,
            output_fields=output_fields,
            search_params=search_params,
        )

        hits = res[0] if res else []
        return [
            VectorSearchResult(
                chunk_id=str(h["id"]),
                distance=float(h["distance"]),
                entity=dict(h.get("entity") or {}),
            )
            for h in hits
        ]