import os
import asyncpg
from pymilvus import MilvusClient

PG_DSN = os.environ["PG_DSN"]
MILVUS_URI = os.environ["MILVUS_URI"]
COL = os.environ["MILVUS_COLLECTION"]

REPO_ID = os.environ["REPO_ID"]
REPO_REF = os.environ["REPO_REF"]
EMB_MODEL = os.environ["EMB_MODEL"]

async def pg_counts():
    conn = await asyncpg.connect(PG_DSN)
    try:
        chunks = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE repo_id=$1 AND ref=$2",
            REPO_ID, REPO_REF
        )
        embedded = await conn.fetchval(
            "SELECT COUNT(*) FROM chunk_embedding_state WHERE embedding_model=$1",
            EMB_MODEL
        )
        return int(chunks or 0), int(embedded or 0)
    finally:
        await conn.close()

def test_postgres_has_chunks_and_embeddings():
    import asyncio
    chunks, embedded = asyncio.get_event_loop().run_until_complete(pg_counts())
    assert chunks > 0, "chunks is empty — indexer likely didn't index repo"
    assert embedded > 0, "chunk_embedding_state is empty — embeddings not recorded"
    assert embedded <= chunks

def test_milvus_has_vectors():
    mc = MilvusClient(uri=MILVUS_URI)
    assert mc.has_collection(COL), f"Milvus collection '{COL}' not found"
    stats = mc.get_collection_stats(COL)
    row_count = int(stats.get("row_count", 0) or 0)
    assert row_count > 0, "Milvus collection is empty"
