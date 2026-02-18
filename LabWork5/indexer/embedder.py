from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import tritonclient.http as httpclient


class EmbedderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Triton HTTP endpoint обычно "localhost:8000"
    # (для tritonclient.http это именно host:port, без "http://")
    url: str = Field(default="localhost:8000", min_length=1, max_length=2048)

    model_name: str = Field(..., min_length=1, max_length=128)
    model_version: str = Field(default="")  # пустая строка = default

    # имена входа/выхода Triton модели
    input_name: str = Field(default="TEXT", min_length=1, max_length=128)
    output_name: str = Field(default="EMBEDDING", min_length=1, max_length=128)

    embedding_dim: int = Field(..., ge=8, le=8192)

    max_batch_size: int = Field(default=64, ge=1, le=2048)

    connection_timeout_s: float = Field(default=10.0, ge=0.1, le=300.0)
    network_timeout_s: float = Field(default=60.0, ge=0.1, le=300.0)

    binary_output: bool = Field(default=True)


class Embedder:
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg

        self._httpclient = httpclient
        self._client = httpclient.InferenceServerClient(
            url=cfg.url,
            verbose=False,
            connection_timeout=cfg.connection_timeout_s,
            network_timeout=cfg.network_timeout_s,
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        out: List[List[float]] = []
        bs = self.cfg.max_batch_size

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            out.extend(self._embed_batch(batch))

        return out

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        arr = np.array(batch, dtype=object).reshape(-1, 1)

        inp = self._httpclient.InferInput(
            self.cfg.input_name,
            arr.shape,
            "BYTES",
        )
        inp.set_data_from_numpy(arr, binary_data=True)

        outputs = [
            self._httpclient.InferRequestedOutput(
                self.cfg.output_name,
                binary_data=self.cfg.binary_output,
            )
        ]

        res = self._client.infer(
            model_name=self.cfg.model_name,
            inputs=[inp],
            outputs=outputs,
            model_version=self.cfg.model_version,
        )

        emb = res.as_numpy(self.cfg.output_name)
        if emb is None:
            raise RuntimeError(f"Triton returned no output '{self.cfg.output_name}'")

        return emb.astype(np.float32).tolist()