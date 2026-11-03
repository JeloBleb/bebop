# bebop — the verified parser

Bebop turns a PDF into the four preprocess assets (`tables.json`, `doc.json`,
`images.json`, markdown) plus an integrity ledger. It is an HTTP
service, not a Python package: a self-contained tree with its own
`requirements.txt` and tests. The design
is a **layout lane** — MinerU proposes regions, prose comes verbatim from the PDF
text layer, and only table regions are transcribed by a vision reader, every
token anchored back to the page's own ink. Nothing invented ships unmarked.

## Why this shape

The **coverage gate** makes the layout proposal total: after it runs, every
text-layer word on a digital page is claimed by exactly one element. A layout
miss degrades — the missed words cluster into a gate element and ship as marked
prose — it can never delete content. **Anchoring** applies the same posture to
tables: a cell's text must match the words under its own region or it is flagged
(`cell_partial` / `cell_unanchored`), and unmatched in-region ink ships on the
table's `unclaimed` side-list. Zero invented data is a property of the
construction rather than a rule anyone enforces.

## Pipeline

| Stage | Module | Does |
|---|---|---|
| ingest | `stages/ingest.py` | PDF → text-layer words and reading-order lines |
| propose | `stages/propose.py` | MinerU layout regions and order (never table content), cached |
| cover | `stages/cover.py` | Snap proposals to their words; cluster uncovered words into gate elements |
| read_table | `stages/read.py`, `stages/anchor.py` | Render each table region, transcribe it, anchor every cell (parallel) |
| compose | `stages/compose.py` | Band merge, spread join, continuation header inheritance |
| emit | `stages/emit.py` | The four assets plus per-value provenance |

`bebop/pipeline.py` runs the sequence; `bebop/model.py` holds the shared types and
`bebop/config.py` every tunable. A read that never parses degrades to prose.

A `doc.json` block's `type` is `heading`, `list`, `paragraph`, `table` or
`image`. The prose roles are MinerU's own calls (its `title` is a `heading`),
never a heuristic, and they annotate rather than filter — every word still ships.
A caption or footnote MinerU attaches to a table or image is a `paragraph`
carrying `caption_of` / `footnote_of`: that table's or image's id, or `null`
when its table degraded to prose.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install "mineru>=3.4,<4"        # or a separate venv + MINERU_BIN
export PREFECT_API_URL=http://<prefect-host>:4200/api   # reach the reader key block
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 5131 --app-dir bebop/service
```

Caches (`cache/{reads,layout}`) anchor on the package tree regardless of CWD. The
reader API key is never on disk: it comes from a Prefect Secret block
(`READER_KEY_BLOCK`). Other knobs: `MINERU_BIN`, `PREFECT_API_URL`,
`READER_API_URL`, `BEBOP_PARSE_CONCURRENCY`.

## HTTP contract

| Endpoint | Request | Response |
|---|---|---|
| `POST /parse` | multipart `file`, optional `document_id`, `pipeline_id` | `{assets, flags, queue, verdicts, telemetry, provenance}` |
| `GET /health` | — | `{ok: true}` |
| `GET /version` | — | service version and reader model |

Parses are serialized by default (`BEBOP_PARSE_CONCURRENCY=1`) — the layout lane
holds one GPU engine.

## Tests

`.venv/bin/python -m pytest bebop/tests -q` from the repo root (offline).

## Pre-warming the read cache

Table reads cache on (model, prompt, image sha), so a bulk parse can be paid for
ahead of time at Gemini Batch Mode's ~50% token price. `batch/` enumerates
the crops a parse would read, submits the cold ones, and writes each response
under the identical cache key; the run that follows is cache-hot. Every command
is resumable and safe to re-run.

```bash
S="--state cache/batch/<run>"; P=".venv/bin/python"   # from the repo root
$P -m bebop.batch $S plan --pdfs /path/to/oms/
$P -m bebop.batch $S submit --dry-run    # files + cost, no spend; drop it to submit
$P -m bebop.batch $S status
$P -m bebop.batch $S collect --wait 24
```
