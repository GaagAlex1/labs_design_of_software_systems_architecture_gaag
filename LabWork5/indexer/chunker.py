import hashlib
from typing import List, Optional, Iterable, Tuple

from pydantic import BaseModel, ConfigDict, Field
from tree_sitter_language_pack import get_parser

from indexer.domain.models import Chunk, Anchor, ChunkKind, Language
from indexer.reader import SourceFile


class ChunkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chars: int = Field(default=1800, ge=200)
    overlap_chars: int = Field(default=200, ge=0)

    min_chars: int = Field(default=200, ge=1)
    max_chunks_per_file: int = Field(default=500, ge=1)

    hard_line_cut_chars: int = Field(default=4000, ge=500)

    normalize_newlines: bool = True


class Chunker:
    def __init__(self, cfg: ChunkerConfig | None = None):
        self.cfg = cfg or ChunkerConfig()
        self._parsers: dict[str, object] = {}

    def _normalize(self, text: str) -> str:
        if not self.cfg.normalize_newlines:
            return text
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _sha1(self, s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def _chunk_id(
        self,
        *,
        repo_id: str,
        commit: str | None,
        path: str,
        start_line: int,
        end_line: int,
        text_hash: str,
    ) -> str:
        base = f"{repo_id}|{commit or ''}|{path}|{start_line}-{end_line}|{text_hash}"
        return self._sha1(base)

    def _split_long_line(self, line: str) -> List[str]:
        n = self.cfg.hard_line_cut_chars
        if len(line) <= n:
            return [line]
        return [line[i : i + n] for i in range(0, len(line), n)]

    def _get_ts_language_name(self, lang: Language, path: str) -> Optional[str]:
        if lang == Language.python:
            return "python"
        if lang == Language.go:
            return "go"
        if lang == Language.javascript:
            return "javascript"
        if lang == Language.typescript:
            if path.lower().endswith(".tsx"):
                return "tsx"
            return "typescript"
        return None

    def _get_parser(self, ts_lang: str):
        if ts_lang in self._parsers:
            return self._parsers[ts_lang]

        p = get_parser(ts_lang)
        self._parsers[ts_lang] = p
        return p

    def _extract_symbol_name(self, node, src_bytes: bytes) -> Optional[str]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        b = src_bytes[name_node.start_byte : name_node.end_byte]
        s = b.decode("utf-8", errors="replace").strip()
        return s or None

    def _iter_def_nodes(self, root, ts_lang: str) -> Iterable:
        if ts_lang == "python":
            wanted = {
                "function_definition",
                "class_definition",
                "decorated_definition",
            }
        elif ts_lang == "go":
            wanted = {
                "function_declaration",
                "method_declaration",
                "type_declaration",
            }
        else:
            wanted = {
                "function_declaration",
                "class_declaration",
                "method_definition",
                "interface_declaration",
                "type_alias_declaration",
                "enum_declaration",
            }

        stack = [root]
        while stack:
            n = stack.pop()
            if n.is_named and n.type in wanted:
                yield n
            for ch in reversed(n.children):
                if ch.is_named:
                    stack.append(ch)

    def _dedup_spans(self, nodes: List) -> List:
        nodes = sorted(nodes, key=lambda n: (n.start_byte, n.end_byte))
        out = []
        last_end = -1
        for n in nodes:
            if n.start_byte >= last_end:
                out.append(n)
                last_end = n.end_byte
        return out

    def _chunk_text_window(
        self,
        *,
        src: SourceFile,
        window_text: str,
        base_start_line: int,
    ) -> list[Chunk]:
        lines = window_text.split("\n")

        expanded_lines: list[str] = []
        for ln in lines:
            expanded_lines.extend(self._split_long_line(ln))

        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_chars = 0

        chunk_start_line = base_start_line
        cur_line = base_start_line - 1

        def flush_chunk(end_line: int) -> None:
            nonlocal buf, buf_chars, chunk_start_line, chunks

            if not buf:
                return

            chunk_text = "\n".join(buf).strip("\n")
            if len(chunk_text) < self.cfg.min_chars:
                buf = []
                buf_chars = 0
                chunk_start_line = end_line + 1
                return

            text_hash = self._sha1(chunk_text)

            repo_id = src.manifest.repo.repo_id
            commit = src.manifest.repo.commit
            path = src.manifest.path

            cid = self._chunk_id(
                repo_id=repo_id,
                commit=commit,
                path=path,
                start_line=chunk_start_line,
                end_line=end_line,
                text_hash=text_hash,
            )

            chunks.append(
                Chunk(
                    chunk_id=cid,
                    repo=src.manifest.repo,
                    path=path,
                    language=src.manifest.language or Language.unknown,
                    anchor=Anchor(start_line=chunk_start_line, end_line=end_line),
                    kind=ChunkKind.block,
                    symbol=None,
                    text=chunk_text,
                    text_hash=text_hash,
                    defines=[],
                    references=[],
                    imports=[],
                    links=dict(src.manifest.source or {}),
                    tags=list(src.manifest.tags),
                    visibility=src.manifest.visibility,
                )
            )

            if self.cfg.overlap_chars > 0:
                tail = chunk_text[-self.cfg.overlap_chars :]
                buf = [tail]
                buf_chars = len(tail)
                chunk_start_line = end_line
            else:
                buf = []
                buf_chars = 0
                chunk_start_line = end_line + 1

        for ln in expanded_lines:
            cur_line += 1
            add_len = len(ln) + (1 if buf else 0)

            if buf and (buf_chars + add_len) > self.cfg.max_chars:
                flush_chunk(end_line=cur_line - 1)
                if len(chunks) >= self.cfg.max_chunks_per_file:
                    break

            buf.append(ln)
            buf_chars += add_len

        if len(chunks) < self.cfg.max_chunks_per_file:
            flush_chunk(end_line=cur_line)

        return chunks

    def _chunk_ast(self, src: SourceFile, ts_lang: str) -> list[Chunk]:
        text = self._normalize(src.text)
        src_bytes = text.encode("utf-8")

        parser = self._get_parser(ts_lang)
        tree = parser.parse(src_bytes)
        root = tree.root_node

        nodes = list(self._iter_def_nodes(root, ts_lang))
        nodes = self._dedup_spans(nodes)

        out: list[Chunk] = []
        for n in nodes:
            start_line = n.start_point[0] + 1
            end_line = n.end_point[0] + 1

            snippet_bytes = src_bytes[n.start_byte : n.end_byte]
            snippet = snippet_bytes.decode("utf-8", errors="replace").strip("\n")
            if not snippet or len(snippet) < self.cfg.min_chars:
                continue

            symbol_name = self._extract_symbol_name(n, src_bytes)
            links = dict(src.manifest.source or {})
            if symbol_name:
                links = {**links, "symbol_name": symbol_name}

            if len(snippet) <= self.cfg.max_chars:
                text_hash = self._sha1(snippet)

                repo_id = src.manifest.repo.repo_id
                commit = src.manifest.repo.commit
                path = src.manifest.path

                cid = self._chunk_id(
                    repo_id=repo_id,
                    commit=commit,
                    path=path,
                    start_line=start_line,
                    end_line=end_line,
                    text_hash=text_hash,
                )

                out.append(
                    Chunk(
                        chunk_id=cid,
                        repo=src.manifest.repo,
                        path=path,
                        language=src.manifest.language or Language.unknown,
                        anchor=Anchor(start_line=start_line, end_line=end_line),
                        kind=ChunkKind.block,
                        symbol=None,
                        text=snippet,
                        text_hash=text_hash,
                        defines=[],
                        references=[],
                        imports=[],
                        links=links,
                        tags=list(src.manifest.tags),
                        visibility=src.manifest.visibility,
                    )
                )
            else:
                out.extend(
                    self._chunk_text_window(
                        src=src,
                        window_text=snippet,
                        base_start_line=start_line,
                    )
                )

            if len(out) >= self.cfg.max_chunks_per_file:
                break

        return out[: self.cfg.max_chunks_per_file]

    def chunk(self, src: SourceFile) -> list[Chunk]:
        text = self._normalize(src.text)

        ts_lang = self._get_ts_language_name(src.manifest.language, src.manifest.path)
        if ts_lang is not None:
            ast_chunks = self._chunk_ast(src, ts_lang)
            if ast_chunks:
                return ast_chunks

        lines = text.split("\n")

        expanded_lines: list[str] = []
        for ln in lines:
            expanded_lines.extend(self._split_long_line(ln))

        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_chars = 0

        chunk_start_line = 1
        cur_line = 0

        def flush_chunk(end_line: int) -> None:
            nonlocal buf, buf_chars, chunk_start_line, chunks

            if not buf:
                return

            chunk_text = "\n".join(buf).strip("\n")
            if len(chunk_text) < self.cfg.min_chars:
                buf = []
                buf_chars = 0
                chunk_start_line = end_line + 1
                return

            text_hash = self._sha1(chunk_text)

            repo_id = src.manifest.repo.repo_id
            commit = src.manifest.repo.commit
            path = src.manifest.path

            cid = self._chunk_id(
                repo_id=repo_id,
                commit=commit,
                path=path,
                start_line=chunk_start_line,
                end_line=end_line,
                text_hash=text_hash,
            )

            chunks.append(
                Chunk(
                    chunk_id=cid,
                    repo=src.manifest.repo,
                    path=path,
                    language=src.manifest.language or Language.unknown,
                    anchor=Anchor(start_line=chunk_start_line, end_line=end_line),
                    kind=ChunkKind.block,
                    symbol=None,
                    text=chunk_text,
                    text_hash=text_hash,
                    defines=[],
                    references=[],
                    imports=[],
                    links=dict(src.manifest.source or {}),
                    tags=list(src.manifest.tags),
                    visibility=src.manifest.visibility,
                )
            )

            if self.cfg.overlap_chars > 0:
                tail = chunk_text[-self.cfg.overlap_chars :]
                buf = [tail]
                buf_chars = len(tail)
                chunk_start_line = end_line
            else:
                buf = []
                buf_chars = 0
                chunk_start_line = end_line + 1

        for ln in expanded_lines:
            cur_line += 1
            add_len = len(ln) + (1 if buf else 0)

            if buf and (buf_chars + add_len) > self.cfg.max_chars:
                flush_chunk(end_line=cur_line - 1)
                if len(chunks) >= self.cfg.max_chunks_per_file:
                    break

            buf.append(ln)
            buf_chars += add_len

        if len(chunks) < self.cfg.max_chunks_per_file:
            flush_chunk(end_line=cur_line)

        return chunks
