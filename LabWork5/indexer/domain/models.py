from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationInfo


class Language(str, Enum):
    python = "python"
    go = "go"
    typescript = "typescript"
    javascript = "javascript"
    json = "json"
    yaml = "yaml"
    toml = "toml"
    markdown = "markdown"
    dockerfile = "dockerfile"
    shell = "shell"
    text = "text"
    unknown = "unknown"


class IndexingProfile(str, Enum):
    default = "default"
    code = "code"
    docs = "docs"
    configs = "configs"
    ci = "ci"


class Visibility(str, Enum):
    public = "public"
    internal = "internal"
    private = "private"


class RepoRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str = Field(..., min_length=1, max_length=200)
    ref: Optional[str] = Field(default=None, max_length=200)
    commit: Optional[str] = Field(default=None, min_length=7, max_length=64)

    @field_validator("commit")
    def _commit_lower(cls, v: str | None) -> str | None:
        return v.lower() if v else v


class FileManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    manifest_version: int = Field(default=1, ge=1)

    repo: RepoRef
    path: str = Field(..., min_length=1, max_length=4096)

    language: Language = Field(default=Language.unknown)
    mime: str | None = Field(default=None, max_length=200)
    size_bytes: int | None = Field(default=None, ge=0)
    content_hash: str | None = Field(default=None, max_length=128)

    indexing_profile: IndexingProfile = Field(default=IndexingProfile.default)
    visibility: Visibility = Field(default=Visibility.internal)

    tags: List[str] = Field(default_factory=list, max_length=128)
    is_generated: Optional[bool] = None

    source: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("path")
    def _normalize_path(cls, v: str) -> str:
        v = v.strip()
        while v.startswith("./"):
            v = v[2:]
        return v


class Anchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)

    @field_validator("end_line")
    def _validate_range(cls, end_line: int, info: ValidationInfo) -> int:
        start_line = info.data.get("start_line")

        if start_line is not None and end_line < start_line:
            raise ValueError("end_line must be >= start_line")

        return end_line


class ChunkKind(str, Enum):
    file = "file"
    module = "module"
    class_ = "class"
    function = "function"
    method = "method"
    block = "block"
    config_section = "config_section"
    doc_section = "doc_section"
    unknown = "unknown"


class SymbolRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=512)
    kind: str | None = Field(default=None, max_length=64)


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., min_length=8, max_length=128)

    repo: RepoRef
    path: str = Field(..., min_length=1, max_length=4096)
    language: Language = Field(default=Language.unknown)

    anchor: Anchor
    kind: ChunkKind = Field(default=ChunkKind.unknown)

    symbol: SymbolRef | None = None

    text: str = Field(..., min_length=1)

    text_hash: str = Field(..., min_length=8, max_length=128)

    defines: List[SymbolRef] = Field(default_factory=list)
    references: List[SymbolRef] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)

    links: Dict[str, str] = Field(default_factory=dict)

    tags: List[str] = Field(default_factory=list, max_length=128)

    visibility: Visibility = Field(default=Visibility.internal)

    @field_validator("path")
    def _normalize_path(cls, v: str) -> str:
        v = v.strip()
        while v.startswith("./"):
            v = v[2:]
        return v


class EmbeddingVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., min_length=8, max_length=128)
    model: str = Field(..., min_length=1, max_length=200)
    dims: int = Field(..., ge=1)
    vector: List[float] = Field(..., min_length=1)

    @field_validator("dims")
    def _dims_match(cls, dims: int, info: ValidationInfo) -> int:
        vec = info.data.get("vector")

        if vec is not None and len(vec) != dims:
            raise ValueError(f"dims ({dims}) must match len(vector) ({len(vec)})")

        return dims


class IndexedFileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: FileManifest
    chunks: List[Chunk] = Field(default_factory=list)
    embeddings: List[EmbeddingVector] = Field(default_factory=list)

