import fnmatch
import hashlib
from pathlib import Path
from typing import Iterator, Optional, List

from pydantic import BaseModel, ConfigDict, Field

from indexer.domain.models import FileManifest, RepoRef, Language, IndexingProfile, Visibility


class RepoReaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_file_size_bytes: int = Field(default=512 * 1024, ge=0)
    compute_content_hash: bool = True

    exclude_globs: List[str] = Field(default_factory=lambda: [
        ".git/**",
        "**/.git/**",
        "**/node_modules/**",
        "**/dist/**",
        "**/build/**",
        "**/target/**",
        "**/.venv/**",
        "**/venv/**",
        "**/__pycache__/**",
        "**/.mypy_cache/**",
        "**/.pytest_cache/**",
        "**/.ruff_cache/**",
        "**/.pytest_cache/**",
        "**/.next/**",
        "**/.turbo/**",
        "**/.cache/**",
        "**/vendor/**",
        "**/coverage/**",
        "**/*.min.js",
        "**/*.min.css",
        "**/*.map",
        "**/*.lock",
    ])

    exclude_exts: List[str] = Field(default_factory=lambda: [
        ".png", ".jpg", ".jpeg", ".gif", ".webp",
        ".pdf", ".zip", ".tar", ".gz", ".7z",
        ".exe", ".dll", ".so", ".dylib",
        ".woff", ".woff2", ".ttf", ".otf",
        ".mp4", ".mov", ".avi", ".mkv",
        ".bin",
    ])


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: FileManifest
    text: str = Field(..., min_length=1)


class Reader:
    def __init__(
        self,
        repo_root: str | Path,
        repo: RepoRef,
        cfg: RepoReaderConfig | None = None,
        *,
        indexing_profile: IndexingProfile = IndexingProfile.default,
        visibility: Visibility = Visibility.internal,
        tags: list[str] | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.repo = repo
        self.cfg = cfg or RepoReaderConfig()
        self.indexing_profile = indexing_profile
        self.visibility = visibility
        self.tags = tags or []

        self._exclude_exts_lower = {e.lower() for e in self.cfg.exclude_exts}

    def _is_excluded(self, rel_posix: str) -> bool:
        return any(fnmatch.fnmatch(rel_posix, pat) for pat in self.cfg.exclude_globs)

    @staticmethod
    def _is_binary(sample: bytes) -> bool:
        if b"\x00" in sample:
            return True

        bad = sum(1 for b in sample if b < 9 or (13 < b < 32))
        return bad > max(8, len(sample) // 20)

    @staticmethod
    def detect_language(rel_path: str) -> Language:
        p = rel_path.replace("\\", "/")
        name = p.split("/")[-1]

        if name == "Dockerfile" or name.startswith("Dockerfile."):
            return Language.dockerfile

        suffix = Path(name).suffix.lower()

        if suffix == ".py":
            return Language.python
        if suffix == ".go":
            return Language.go
        if suffix in (".ts", ".tsx"):
            return Language.typescript
        if suffix in (".js", ".jsx", ".mjs", ".cjs"):
            return Language.javascript
        if suffix == ".md":
            return Language.markdown
        if suffix == ".json":
            return Language.json
        if suffix in (".yml", ".yaml"):
            return Language.yaml
        if suffix == ".toml":
            return Language.toml
        if suffix in (".sh", ".bash", ".zsh"):
            return Language.shell

        if name in (".env", ".gitignore", ".dockerignore"):
            return Language.text

        return Language.unknown

    def read_text_file(self, path: Path) -> Optional[str]:
        with path.open("rb") as f:
            sample = f.read(4096)
            if self._is_binary(sample):
                return None
            data = sample + f.read()

        return data.decode("utf-8")

    def iter_repo_paths(self) -> Iterator[Path]:
        for p in self.repo_root.rglob("*"):
            if not p.is_file():
                continue

            rel = p.relative_to(self.repo_root).as_posix()

            if self._is_excluded(rel):
                continue

            if p.suffix.lower() in self._exclude_exts_lower:
                continue

            size = p.stat().st_size

            if size <= 0 or size > self.cfg.max_file_size_bytes:
                continue

            yield p

    def read_repo(self) -> list[SourceFile]:
        out: list[SourceFile] = []

        for abs_path in self.iter_repo_paths():
            rel_path = abs_path.relative_to(self.repo_root).as_posix()
            text = self.read_text_file(abs_path)
            if not text:
                continue

            lang = self.detect_language(rel_path)

            normalized = text.replace("\r\n", "\n").replace("\r", "\n")
            content_hash = None
            if self.cfg.compute_content_hash:
                content_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()

            size_bytes = len(normalized.encode("utf-8"))

            manifest = FileManifest(
                repo=self.repo,
                path=rel_path,
                language=lang,
                size_bytes=size_bytes,
                content_hash=content_hash,
                indexing_profile=self.indexing_profile,
                visibility=self.visibility,
                tags=list(self.tags),
                source={"kind": "local_fs", "abs_path": str(abs_path)},
            )

            out.append(SourceFile(manifest=manifest, text=text))

        return out