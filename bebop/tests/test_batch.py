"""The Gemini Batch pre-warm driver (bebop/batch, issue #97) — fake
transport, mocked layout lane, no network, no GPU, no spend.

The load-bearing assertions: the driver enumerates the SAME PNG bytes a
parse reads (so the same cache key), the request line carries the same
prompt/image/knobs the interactive body does, a collected response is a
cache HIT for read_png, and every command is idempotent across a reload
of the state file.
"""

import base64
import json

import pytest

from bebop import config, pipeline
from bebop.stages import read as reader_module
from bebop.batch import driver, request, state as state_module
from bebop.batch.crops import enumerate_crops
from bebop.stages.read import read_png


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "READ_CACHE_DIR", str(tmp_path / "reads"))
    monkeypatch.setattr(config, "READER_RETRY_SLEEP_S", 0.0)
    monkeypatch.setattr(reader_module, "_cooloff", reader_module._CoolOff())


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "batch"


TABLE = [
    (72, 100, "Item Amount"),
    (72, 120, "Rent 1,000"),
    (72, 140, "NOI 900"),
]
OTHER = [(72, 100, "Unit Rate"), (72, 120, "A 12")]


@pytest.fixture
def two_pdfs(make_pdf, element, monkeypatch):
    """Two one-table documents behind a mocked layout lane: propose()
    returns one table proposal per page, the gate snaps it as it would
    a MinerU proposal."""
    a = make_pdf(TABLE, name="a.pdf")
    b = make_pdf(OTHER, name="b.pdf")

    def fake_propose(pdf_path):
        return [element(type="table", bbox=(60, 80, 400, 160), order=0, header_rows=1)]

    monkeypatch.setattr(pipeline, "propose", fake_propose)
    return a, b


def good_line(key, text='{"header_rows":1,"rows":[["Item","Amount"],["Rent","1,000"],["NOI","900"]]}'):
    return {
        "key": key,
        "response": {
            "candidates": [{"content": {"role": "model", "parts": [
                {"text": "thinking...", "thought": True},
                {"text": text},
            ]}}],
            "usageMetadata": {"promptTokenCount": 812, "candidatesTokenCount": 44,
                              "thoughtsTokenCount": 300, "totalTokenCount": 1156},
        },
    }


class FakeApi:
    """The GeminiBatchApi surface, recording every call. Result lines
    come from `responses` (key -> line) or the good_line default."""

    def __init__(self, responses=None, state="BATCH_STATE_SUCCEEDED"):
        self.responses = responses or {}
        self.state = state
        self.uploaded = {}  # file name -> keys in that file
        self.created = []
        self.gets = 0
        self.downloads = []

    def upload_jsonl(self, path, display_name):
        keys = [json.loads(ln)["key"] for ln in path.read_text().splitlines() if ln]
        name = f"files/up-{len(self.uploaded)}"
        self.uploaded[name] = keys
        return name

    def create_batch(self, model, file_name, display_name):
        self.created.append((model, file_name, display_name))
        return {"name": f"batches/{len(self.created)}",
                "metadata": {"state": "BATCH_STATE_PENDING", "inputConfig": {"fileName": file_name}}}

    def get_batch(self, name):
        self.gets += 1
        idx = int(name.split("/")[1]) - 1
        file_name = self.created[idx][1]
        op = {"name": name, "metadata": {"state": self.state}}
        if self.state == "BATCH_STATE_SUCCEEDED":
            op["done"] = True
            op["response"] = {"responsesFile": f"files/out-{file_name.split('-')[1]}"}
        return op

    def download_lines(self, file_name):
        self.downloads.append(file_name)
        keys = self.uploaded[f"files/up-{file_name.split('-')[1]}"]
        for k in keys:
            yield self.responses.get(k) or good_line(k)


# --- parity -------------------------------------------------------------------


def test_driver_enumerates_exactly_the_pngs_a_parse_reads(two_pdfs):
    a, b = two_pdfs
    seen = []

    def capturing(png):
        seen.append(png)
        return '{"header_rows":1,"rows":[["Item","Amount"],["Rent","1,000"],["NOI","900"]]}', 0.001

    pipeline.parse(a, transport=capturing)
    pipeline.parse(b, transport=capturing)
    assert seen  # the parses actually read tables

    crops = list(enumerate_crops([a, b]))
    assert len(crops) == 2
    assert {c.png for c in crops} == set(seen)
    for c in crops:
        # the driver's key IS the reader's key, computed by the reader
        assert c.cache_path == reader_module._cache_path(c.png)
        assert c.cache_path == reader_module._cache_path_for_sha(c.image_sha)
        entry = json.loads(c.cache_path.read_text())  # the parse wrote it
        assert entry["image_sha256"] == c.image_sha
        assert c.cached


def test_jsonl_line_carries_the_interactive_payload(two_pdfs):
    (crop,) = list(enumerate_crops([two_pdfs[0]]))
    line = json.loads(request.jsonl_line(crop.image_sha, crop.png))
    assert line["key"] == crop.image_sha
    req = line["request"]
    (turn,) = req["contents"]
    assert turn["role"] == "user"
    image, text = turn["parts"]  # image first, prompt second — the interactive order
    assert image["inlineData"]["mimeType"] == "image/png"
    assert base64.standard_b64decode(image["inlineData"]["data"]) == crop.png
    assert text == {"text": reader_module.PROMPT}
    assert req["generationConfig"] == {
        "temperature": config.READER_TEMPERATURE,
        "maxOutputTokens": config.READER_MAX_TOKENS,
    }
    assert set(req) == {"contents", "generationConfig"}  # nothing the interactive body lacks
    assert request.wire_model() == "gemini-3.7-flash"


def test_response_text_matches_the_compat_content_and_usage():
    text, usage = request.response_text(good_line("k")["response"])
    assert text == '{"header_rows":1,"rows":[["Item","Amount"],["Rent","1,000"],["NOI","900"]]}'
    assert usage == {"cost": 0.0, "prompt_tokens": 812, "completion_tokens": 44}
    assert request.response_text({"candidates": []}) == (None, {"cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0})


# --- plan / dry run -----------------------------------------------------------


def test_plan_skips_cached_crops_and_writes_the_rest(two_pdfs, state_dir):
    a, b = two_pdfs
    # prime a's crop the interactive way; b stays cold
    (crop_a,) = list(enumerate_crops([a]))
    read_png(crop_a.png, transport=lambda png: ('{"rows":[["x"]]}', 0.001))

    st = driver.load_or_fresh(state_dir)
    rep = driver.plan([a, b], st, state_dir)
    assert rep["crops"] == 2 and rep["already_cached"] == 1 and rep["new_requests"] == 1
    assert rep["new_chunks"] == ["requests-000.jsonl"]
    assert rep["estimated_cost_usd"] == pytest.approx(0.012)
    lines = (state_dir / "requests-000.jsonl").read_text().splitlines()
    assert len(lines) == 1
    (crop_b,) = list(enumerate_crops([b]))
    assert json.loads(lines[0])["key"] == crop_b.image_sha
    assert st.crops[crop_b.image_sha].status == state_module.PENDING
    assert crop_a.image_sha not in st.crops

    # re-planning adds nothing and rolls no new chunk
    rep2 = driver.plan([a, b], st, state_dir)
    assert rep2["new_requests"] == 0 and rep2["already_planned"] == 1 and rep2["new_chunks"] == []


def test_plan_dedupes_identical_crops_across_documents(two_pdfs, make_pdf, state_dir):
    a, _ = two_pdfs
    a2 = make_pdf(TABLE, name="a-copy.pdf")  # byte-identical table region
    st = driver.load_or_fresh(state_dir)
    rep = driver.plan([a, a2], st, state_dir)
    assert rep["crops"] == 2 and rep["duplicate_in_run"] == 1 and rep["new_requests"] == 1


def test_chunks_roll_at_the_byte_limit(two_pdfs, state_dir):
    st = driver.load_or_fresh(state_dir)
    rep = driver.plan(two_pdfs, st, state_dir, chunk_bytes=1)  # one request per chunk
    assert rep["new_chunks"] == ["requests-000.jsonl", "requests-001.jsonl"]
    assert [j.requests for j in st.jobs] == [1, 1]


def test_dry_run_submit_touches_no_api(two_pdfs, state_dir):
    st = driver.load_or_fresh(state_dir)
    driver.plan(two_pdfs, st, state_dir)

    class Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"API called in dry run: {name}")

    rep = driver.submit(st, state_dir, Exploding(), dry_run=True)
    assert rep["dry_run"] and rep["requests"] == 2
    assert rep["estimated_cost_usd"] == pytest.approx(0.024)
    assert rep["jobs"] == [] and all(j.name is None for j in st.jobs)
    assert all(r.status == state_module.PENDING for r in st.crops.values())


# --- submit / collect ---------------------------------------------------------


def test_collect_writes_entries_the_reader_serves_as_hits(two_pdfs, state_dir):
    api = FakeApi()
    st = driver.load_or_fresh(state_dir)
    driver.plan(two_pdfs, st, state_dir)
    rep = driver.submit(st, state_dir, api)
    assert rep["jobs"] == ["batches/1"]
    assert api.created[0][0] == "gemini-3.7-flash"  # the reader's wire model
    assert all(r.status == state_module.SUBMITTED for r in st.crops.values())

    rep = driver.collect(st, state_dir, api)
    assert rep["written"] == 2 and rep["unparsed"] == 0 and rep["failed"] == 0

    def exploding(png):
        raise AssertionError("transport called: the batch entry was not a hit")

    for crop in enumerate_crops(two_pdfs):
        assert crop.cached
        r = read_png(crop.png, transport=exploding)
        assert r.cached and r.rows[0] == ("Item", "Amount") and r.header_rows == 1
        entry = json.loads(crop.cache_path.read_text())
        assert entry["image_sha256"] == crop.image_sha
        assert entry["model"] == config.READER_MODEL
        assert entry["prompt_sha256"] == reader_module._sha(reader_module.PROMPT.encode())
        assert entry["error"] is None and entry["cost"] == 0.0
        assert entry["prompt_tokens"] == 812 and entry["completion_tokens"] == 44

    # schema parity: the same key set an interactive read writes
    interactive = reader_module._cache_path(b"other")
    read_png(b"other", transport=lambda png: ('{"rows":[["x"]]}', {"prompt_tokens": 1, "completion_tokens": 1}))
    assert set(json.loads(interactive.read_text())) == set(entry)


def test_resume_from_the_state_file(two_pdfs, state_dir):
    api = FakeApi(state="BATCH_STATE_RUNNING")
    st = driver.load_or_fresh(state_dir)
    driver.plan(two_pdfs, st, state_dir)
    driver.submit(st, state_dir, api)
    del st

    # a later process: nothing in memory, only state.json on disk
    st2 = driver.load_or_fresh(state_dir)
    assert [j.name for j in st2.jobs] == ["batches/1"]
    assert driver.status(st2, state_dir, api)["jobs"][0]["state"] == "BATCH_STATE_RUNNING"
    rep = driver.collect(st2, state_dir, api)
    assert rep["written"] == 0 and rep["jobs_pending"] == ["batches/1"]
    assert api.downloads == []

    api.state = "BATCH_STATE_SUCCEEDED"
    st3 = driver.load_or_fresh(state_dir)
    rep = driver.collect(st3, state_dir, api)
    assert rep["written"] == 2 and api.downloads == ["files/out-0"]

    # idempotent: a re-run collects nothing twice and downloads nothing twice
    st4 = driver.load_or_fresh(state_dir)
    rep = driver.collect(st4, state_dir, api)
    assert rep["written"] == 0 and rep["jobs_pending"] == [] and api.downloads == ["files/out-0"]
    assert driver.submit(st4, state_dir, api)["requests"] == 0  # nothing left to submit
    # and a re-plan sees every crop primed
    rep = driver.plan(two_pdfs, st4, state_dir)
    assert rep["already_cached"] == 2 and rep["new_requests"] == 0


def test_unparseable_or_failed_responses_leave_the_cache_cold(two_pdfs, state_dir):
    crops = list(enumerate_crops(two_pdfs))
    bad, err = crops[0].image_sha, crops[1].image_sha
    api = FakeApi(responses={
        bad: good_line(bad, text="I could not read this table."),
        err: {"key": err, "error": {"code": 13, "message": "internal"}},
    })
    st = driver.load_or_fresh(state_dir)
    driver.plan(two_pdfs, st, state_dir)
    driver.submit(st, state_dir, api)
    rep = driver.collect(st, state_dir, api)
    assert rep == {**rep, "written": 0, "unparsed": 1, "failed": 1}
    assert not crops[0].cached and not crops[1].cached
    assert st.crops[bad].status == state_module.UNPARSED
    assert st.crops[bad].error == "no JSON object in response"
    assert st.crops[err].status == state_module.FAILED

    # a re-plan re-queues both into a fresh chunk — the cache is still cold
    rep = driver.plan(two_pdfs, st, state_dir)
    assert rep["new_requests"] == 2 and rep["new_chunks"] == ["requests-001.jsonl"]


def test_collect_never_overwrites_a_primed_entry(two_pdfs, state_dir):
    api = FakeApi()
    st = driver.load_or_fresh(state_dir)
    driver.plan(two_pdfs, st, state_dir)
    driver.submit(st, state_dir, api)
    # an interactive read lands first (a parse ran while the batch was queued)
    (crop,) = list(enumerate_crops([two_pdfs[0]]))
    read_png(crop.png, transport=lambda png: ('{"rows":[["interactive"]]}', 0.001))
    rep = driver.collect(st, state_dir, api)
    assert rep["written"] == 1 and rep["already_cached"] == 1
    assert read_png(crop.png, transport=None).rows == (("interactive",),)


def test_state_from_another_model_or_prompt_is_not_resumable(two_pdfs, state_dir, monkeypatch):
    st = driver.load_or_fresh(state_dir)
    driver.plan(two_pdfs, st, state_dir)
    monkeypatch.setattr(config, "READER_MODEL", "other/model")
    with pytest.raises(RuntimeError, match="start a new run"):
        driver.load_or_fresh(state_dir)


def test_state_dir_is_anchored_on_the_service_tree():
    assert state_module.default_state_dir() == config._SERVICE_ROOT / "cache" / "batch"
