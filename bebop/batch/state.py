"""Resumable state file, one JSON document per pre-warm run: crop -> request
mapping by image sha, chunk membership, job state, and per-crop outcome."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import config
from ..stages import read as reader

VERSION = 1

# crop statuses
PENDING = "pending"  # enumerated, in a chunk, no job yet
SUBMITTED = "submitted"  # its chunk has a batch job
WRITTEN = "written"  # response landed in the read cache
UNPARSED = "unparsed"  # response came back but never parsed — cache left cold
FAILED = "failed"  # the batch reported a per-request error — cache left cold


@dataclass
class CropRecord:
    pdf: str
    page: int
    reading_order: int
    chunk: str  # chunk file name (relative to the state dir)
    status: str = PENDING
    error: str | None = None


@dataclass
class Job:
    chunk: str
    name: str | None = None  # batches/... once submitted
    file_name: str | None = None  # files/... the upload
    requests: int = 0
    bytes: int = 0
    state: str | None = None
    responses_file: str | None = None
    collected: bool = False


@dataclass
class State:
    model: str
    prompt_sha256: str
    cache_dir: str
    crops: dict[str, CropRecord] = field(default_factory=dict)  # image sha -> record
    jobs: list[Job] = field(default_factory=list)
    version: int = VERSION

    @classmethod
    def fresh(cls) -> "State":
        return cls(
            model=reader._cache_model(),
            prompt_sha256=reader._sha(reader.PROMPT.encode()),
            cache_dir=str(reader._cache_path_for_sha("x").parent),
        )

    def check_current(self) -> None:
        """A state file from another (model, prompt) is not resumable:
        its keys would land in a different cache dir than the one the
        reader consults today."""
        now = State.fresh()
        if (self.model, self.prompt_sha256) != (now.model, now.prompt_sha256):
            raise RuntimeError(
                f"state was made for model={self.model!r} prompt={self.prompt_sha256[:12]} "
                f"but the reader now keys model={now.model!r} prompt={now.prompt_sha256[:12]}; "
                f"start a new run"
            )

    def job_for(self, chunk: str) -> Job | None:
        return next((j for j in self.jobs if j.chunk == chunk), None)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1))
        tmp.replace(path)  # atomic: a killed run never leaves a torn file

    @classmethod
    def load(cls, path: Path) -> "State":
        d = json.loads(path.read_text())
        if d.get("version") != VERSION:
            raise RuntimeError(f"unsupported state version {d.get('version')} in {path}")
        return cls(
            model=d["model"], prompt_sha256=d["prompt_sha256"], cache_dir=d["cache_dir"],
            crops={k: CropRecord(**v) for k, v in d["crops"].items()},
            jobs=[Job(**j) for j in d["jobs"]],
        )

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.crops.values():
            out[r.status] = out.get(r.status, 0) + 1
        return out


def default_state_dir() -> Path:
    """Beside the read cache, anchored on the service tree like every
    other cache path (config._SERVICE_ROOT), never the CWD."""
    return config._SERVICE_ROOT / "cache" / "batch"
