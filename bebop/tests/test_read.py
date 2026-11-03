"""read_png transport/cache behavior (ported from
v2/tests/test_reader.py) — fake transports, no network. The render test
adapts to v3's element-based rendering; everything else is the v2 catalog
verbatim."""

import hashlib
import io
import json

import pytest

from bebop import config
from bebop.stages import read as reader_module
from bebop.stages.read import read_png, render
from bebop.stages.ingest import ingest
from bebop.stages.read import TableUnread, read_table


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "READ_CACHE_DIR", str(tmp_path / "reads"))
    monkeypatch.setattr(config, "READER_RETRY_SLEEP_S", 0.0)


@pytest.fixture(autouse=True)
def isolated_key_cache(monkeypatch):
    # _key() memoizes for the process lifetime (bebop is a long-running
    # service); tests need a clean slate each time regardless of load order.
    # The host env may legitimately set the block-name overrides (unit env,
    # worker envs) — clear both spellings so assertions see config defaults.
    monkeypatch.setattr(reader_module, "_key_cache", None)
    monkeypatch.delenv("READER_KEY_BLOCK", raising=False)
    monkeypatch.delenv("OPENROUTER_KEY_BLOCK", raising=False)


@pytest.fixture(autouse=True)
def fresh_cooloff(monkeypatch):
    # the cool-off latch is module state shared by all reader threads; a
    # test that trips it must not delay the next test's reads.
    monkeypatch.setattr(reader_module, "_cooloff", reader_module._CoolOff())


def transport_returning(*texts):
    """A fake transport that serves each text once, then fails the test."""
    remaining = list(texts)

    def _t(png):
        assert remaining, "transport called more often than expected"
        return remaining.pop(0), 0.001

    return _t


def test_clean_minified_response_parses(tmp_path):
    r = read_png(b"png-1", transport=transport_returning('{"rows":[["Total","1,234"],["NOI",null]]}'))
    assert r.rows == (("Total", "1,234"), ("NOI", None))
    assert r.error is None and not r.cached
    assert r.cost == pytest.approx(0.001)


def test_second_read_is_served_from_cache():
    read_png(b"png-2", transport=transport_returning('{"rows":[["a"]]}'))
    exploding = transport_returning()  # any call fails the test
    r = read_png(b"png-2", transport=exploding)
    assert r.cached and r.rows == (("a",),) and r.cost == 0.0


def test_json_embedded_in_prose_is_extracted():
    r = read_png(b"png-3", transport=transport_returning('Here you go:\n{"rows":[["x"]]}\nDone!'))
    assert r.rows == (("x",),) and r.error is None


def test_transport_failure_every_attempt_yields_error_no_rows():
    def failing(png):
        raise OSError("connection refused")

    r = read_png(b"png-5", transport=failing)
    assert r.rows is None
    assert "connection refused" in r.error


def test_cached_transport_failure_is_retried_not_poisonous():
    def failing(png):
        raise OSError("connection refused")

    r1 = read_png(b"png-9", transport=failing)
    assert r1.rows is None and not r1.cached
    # the failure entry is an audit record, not a result: the next read
    # attempts the transport again and the region recovers
    r2 = read_png(b"png-9", transport=transport_returning('{"rows":[["ok"]]}'))
    assert not r2.cached and r2.rows == (("ok",),)
    # and the recovery is what the cache serves from then on
    r3 = read_png(b"png-9", transport=transport_returning())
    assert r3.cached and r3.rows == (("ok",),)


def test_cache_key_depends_on_model(monkeypatch):
    read_png(b"png-6", transport=transport_returning('{"rows":[["from-flash"]]}'))
    monkeypatch.setattr(config, "READER_MODEL", "other/model")
    r = read_png(b"png-6", transport=transport_returning('{"rows":[["from-other"]]}'))
    assert not r.cached and r.rows == (("from-other",),)


def test_render_is_deterministic_and_empty_clip_is_none(make_pdf, element):
    path = make_pdf([(72, 100, "Revenue 1,234 5,678")])
    el = element(type="table", bbox=(60, 80, 400, 120))
    png = render(path, el)
    assert png == render(path, el)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    off_page = element(type="table", bbox=(-500, -500, -400, -450))
    assert render(path, off_page) is None


def test_scalar_row_is_wrapped_not_crashed():
    r = read_png(b"png-7", transport=transport_returning('{"rows":["loose text line"]}'))
    assert r.rows == (("loose text line",),)


def fake_secret(value=None, error=None):
    """Stand-in for ``prefect.blocks.system.Secret`` — records each block
    name ``_key()`` asked for and returns a canned value or raises,
    without a real Prefect server (bebop/reader.py's ``_key()``)."""
    calls = []

    class _Block:
        def get(self):
            return value

    class _Secret:
        @staticmethod
        def load(name):
            calls.append(name)
            if error is not None:
                raise error
            return _Block()

    _Secret.calls = calls
    return _Secret


def test_key_loads_the_pinned_block(monkeypatch):
    fake = fake_secret(value="sk-or-test")
    monkeypatch.setattr(reader_module, "Secret", fake)
    assert reader_module._key() == "sk-or-test"
    assert fake.calls == [config.READER_KEY_BLOCK]


def test_key_is_cached_for_the_process_lifetime(monkeypatch):
    fake = fake_secret(value="sk-or-test")
    monkeypatch.setattr(reader_module, "Secret", fake)
    reader_module._key()
    reader_module._key()
    assert fake.calls == [config.READER_KEY_BLOCK]  # loaded once, not twice


def test_key_block_name_is_overridable_via_env(monkeypatch):
    # the pre-#95 spelling the :5131 unit env and worker envs still set
    monkeypatch.setenv("OPENROUTER_KEY_BLOCK", "openrouter-api-key-dev")
    fake = fake_secret(value="sk-or-test")
    monkeypatch.setattr(reader_module, "Secret", fake)
    reader_module._key()
    assert fake.calls == ["openrouter-api-key-dev"]


def test_reader_key_block_env_wins_over_the_legacy_spelling(monkeypatch):
    monkeypatch.setenv("OPENROUTER_KEY_BLOCK", "openrouter-api-key-dev")
    monkeypatch.setenv("READER_KEY_BLOCK", "gemini-api-key-prod")
    fake = fake_secret(value="AIza-test")
    monkeypatch.setattr(reader_module, "Secret", fake)
    reader_module._key()
    assert fake.calls == ["gemini-api-key-prod"]


def test_key_raises_a_clear_error_when_the_block_is_missing(monkeypatch):
    fake = fake_secret(error=RuntimeError("ObjectNotFound"))
    monkeypatch.setattr(reader_module, "Secret", fake)
    with pytest.raises(RuntimeError, match=config.READER_KEY_BLOCK):
        reader_module._key()


# --- throttle handling (issue #93: 392/402 table_unread flags were 429) ----


class FakeTime:
    """Stands in for reader's `time` module: sleeps are recorded and
    advance a fake monotonic clock, so latch waits and backoffs are
    asserted exactly, with no real sleeping."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_time(monkeypatch):
    ft = FakeTime()
    monkeypatch.setattr(reader_module, "time", ft)
    return ft


def http_error(code, headers=None):
    import urllib.error

    msgs = {429: "Too Many Requests", 503: "Service Unavailable"}
    return urllib.error.HTTPError(
        "https://openrouter.test/api/v1/chat/completions",
        code, msgs.get(code, "boom"), headers or {}, None,
    )


def throttled_then(text, failures, code=429, headers=None):
    """A transport raising `failures` throttle-class errors, then `text`."""
    calls = []

    def _t(png):
        calls.append(1)
        if len(calls) <= failures:
            raise http_error(code, headers=headers)
        return text, 0.001

    _t.calls = calls
    return _t


def test_throttle_class_gets_backoff_not_content_retries(fake_time, monkeypatch):
    # a content retry would announce itself by this sentinel sleep
    monkeypatch.setattr(config, "READER_RETRY_SLEEP_S", 777.0)
    flaky = throttled_then('{"rows":[["ok"]]}', failures=3)

    r = read_png(b"png-throttle-1", transport=flaky)
    assert r.rows == (("ok",),) and r.error is None

    # 4 transport calls — more than READER_RETRIES could ever make — all
    # inside ONE content attempt: the throttle budget is its own
    assert len(flaky.calls) == 4
    assert 777.0 not in fake_time.sleeps

    # one jittered exponential backoff per throttle failure
    assert len(fake_time.sleeps) == 3
    for k, s in enumerate(fake_time.sleeps):
        base = min(config.READER_THROTTLE_CAP_S, config.READER_THROTTLE_BASE_S * 2 ** k)
        assert 0.5 * base <= s <= 1.5 * base

    # and the waits were latch-mediated: the final backoff ended exactly
    # when the shared cool-off reopened
    assert reader_module._cooloff._resume_at == pytest.approx(fake_time.now)


def test_retry_after_is_honored(fake_time):
    flaky = throttled_then('{"rows":[["ok"]]}', failures=1,
                           headers={"Retry-After": "45"})
    r = read_png(b"png-throttle-2", transport=flaky)
    assert r.rows == (("ok",),)
    # max(45, first jittered backoff of 1-3s) = the server's 45
    assert fake_time.sleeps == [45.0]


def test_one_throttled_thread_pauses_every_read(fake_time):
    # a sibling reader thread hit a 429 and tripped the shared latch
    # (exactly what _throttled does); this read — different table, clean
    # transport — must wait the window out before its first call
    reader_module._cooloff.trip(30.0)
    r = read_png(b"png-throttle-3", transport=transport_returning('{"rows":[["ok"]]}'))
    assert r.rows == (("ok",),)
    assert fake_time.sleeps == [30.0]


def test_5xx_is_throttle_class(fake_time):
    flaky = throttled_then('{"rows":[["ok"]]}', failures=2, code=503)
    r = read_png(b"png-throttle-4", transport=flaky)
    assert r.rows == (("ok",),)
    assert len(flaky.calls) == 3 and len(fake_time.sleeps) == 2


def test_throttle_classification():
    assert reader_module._throttle_class(http_error(429))
    assert reader_module._throttle_class(http_error(500))
    assert reader_module._throttle_class(http_error(599))
    assert not reader_module._throttle_class(http_error(404))
    assert not reader_module._throttle_class(OSError("connection refused"))


def test_non_throttle_http_error_keeps_content_retry_path(fake_time, monkeypatch):
    monkeypatch.setattr(config, "READER_RETRY_SLEEP_S", 7.0)
    calls = []

    def failing(png):
        calls.append(1)
        raise http_error(404)

    r = read_png(b"png-throttle-5", transport=failing)
    assert r.rows is None and "HTTP Error 404" in r.error
    assert len(calls) == config.READER_RETRIES  # no throttle envelope
    assert fake_time.sleeps == [7.0, 7.0]  # today's fixed sleep, no backoff


def test_unparseable_json_keeps_quick_content_retries(fake_time, monkeypatch):
    monkeypatch.setattr(config, "READER_RETRY_SLEEP_S", 7.0)
    # transport_returning fails the test on a 4th call: READER_RETRIES exactly
    r = read_png(b"png-throttle-6",
                 transport=transport_returning("garbage", "garbage", "garbage"))
    assert r.rows is None and r.error == "no JSON object in response"
    assert fake_time.sleeps == [7.0, 7.0]


def test_exhausted_throttle_budget_records_error_and_degrades(fake_time):
    calls = []

    def always_throttled(png):
        calls.append(1)
        raise http_error(429)

    r = read_png(b"png-throttle-7", transport=always_throttled)
    # rows=None is the degrade signal — read_table raises TableUnread and
    # the element ships as text-layer prose, flag text unchanged
    assert r.rows is None
    assert "HTTP Error 429: Too Many Requests" in r.error
    # each content attempt carried one full throttle envelope
    assert len(calls) == config.READER_RETRIES * (config.READER_THROTTLE_RETRIES + 1)
    backoffs = [s for s in fake_time.sleeps if s > 0]
    assert len(backoffs) == config.READER_RETRIES * config.READER_THROTTLE_RETRIES


# --- transport generalization (issue #95: OpenRouter -> any OpenAI-compat
# endpoint; Google Gemini API flip parked on the gemini-api-key-prod block) --


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def fake_urlopen(monkeypatch, response: dict):
    """Serve one canned JSON response and capture the Request that
    _http_transport built — the real transport exercised, no network."""
    captured = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _open(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["auth"] = req.get_header("Authorization")
        captured["timeout"] = timeout
        return _Resp(json.dumps(response).encode())

    monkeypatch.setattr(reader_module.urllib.request, "urlopen", _open)
    return captured


def test_http_transport_openrouter_shape(monkeypatch):
    # today's default: prefixed model on the wire, usage.cost in dollars
    monkeypatch.setattr(config, "READER_API_URL", OPENROUTER_URL)
    monkeypatch.setattr(reader_module, "Secret", fake_secret(value="sk-or-key"))
    captured = fake_urlopen(monkeypatch, {
        "choices": [{"message": {"content": '{"rows":[["x"]]}'}}],
        "usage": {"prompt_tokens": 812, "completion_tokens": 44, "cost": 0.00123},
    })
    text, usage = reader_module._http_transport(b"png-bytes")
    assert captured["url"] == OPENROUTER_URL
    assert captured["body"]["model"] == "google/gemini-3.7-flash"  # nothing stripped
    assert captured["auth"] == "Bearer sk-or-key"
    assert text == '{"rows":[["x"]]}'
    assert usage == {"cost": pytest.approx(0.00123),
                     "prompt_tokens": 812, "completion_tokens": 44}


def test_http_transport_google_shape(monkeypatch):
    # post-flip: bare model on the wire, token counts only — no usage.cost
    monkeypatch.setattr(config, "READER_API_URL", GOOGLE_URL)
    monkeypatch.setattr(reader_module, "Secret", fake_secret(value="AIza-key"))
    captured = fake_urlopen(monkeypatch, {
        "choices": [{"message": {"content": '{"rows":[["x"]]}'}}],
        "usage": {"prompt_tokens": 812, "completion_tokens": 44, "total_tokens": 856},
    })
    text, usage = reader_module._http_transport(b"png-bytes")
    assert captured["url"] == GOOGLE_URL
    assert captured["body"]["model"] == "gemini-3.7-flash"  # google/ stripped
    assert text == '{"rows":[["x"]]}'
    assert usage == {"cost": 0.0, "prompt_tokens": 812, "completion_tokens": 44}


def test_wire_model_strips_the_prefix_only_for_the_google_endpoint(monkeypatch):
    monkeypatch.setattr(config, "READER_API_URL", OPENROUTER_URL)
    assert reader_module._wire_model() == "google/gemini-3.7-flash"
    monkeypatch.setattr(config, "READER_API_URL", GOOGLE_URL)
    assert reader_module._wire_model() == "gemini-3.7-flash"
    monkeypatch.setattr(config, "READER_MODEL", "gemini-3.7-flash")
    assert reader_module._wire_model() == "gemini-3.7-flash"  # bare stays bare


def test_cache_key_both_model_spellings_hit_the_primed_dir(monkeypatch):
    # The dir name a primed corpus cache actually has on disk: the pre-#95
    # formula applied to the PREFIXED spelling. Computed here independently
    # of _cache_model as the fixed oracle the normalization must preserve.
    primed_ctx = hashlib.sha256(
        f"google/gemini-3.7-flash\n{reader_module.PROMPT}".encode()
    ).hexdigest()[:12]
    primed_dir = f"google_gemini-3.7-flash-{primed_ctx}"

    monkeypatch.setattr(config, "READER_MODEL", "google/gemini-3.7-flash")
    prefixed = reader_module._cache_path(b"img")
    monkeypatch.setattr(config, "READER_MODEL", "gemini-3.7-flash")
    bare = reader_module._cache_path(b"img")

    assert prefixed == bare  # one model, one cache dir — either spelling
    assert prefixed.parent.name == primed_dir  # and it is the existing dir


def test_primed_cache_replays_across_the_spelling_flip(monkeypatch):
    monkeypatch.setattr(config, "READER_MODEL", "google/gemini-3.7-flash")
    read_png(b"png-flip", transport=transport_returning('{"rows":[["primed"]]}'))
    monkeypatch.setattr(config, "READER_MODEL", "gemini-3.7-flash")
    r = read_png(b"png-flip", transport=transport_returning())  # any call fails the test
    assert r.cached and r.rows == (("primed",),) and r.cost == 0.0


def test_pre95_cache_entry_without_token_fields_still_loads():
    # a corpus entry written before #95 carries no token fields — additive
    # schema only, so it must replay exactly as it always did
    path = reader_module._cache_path(b"png-old")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model": config.READER_MODEL,
        "prompt_sha256": reader_module._sha(reader_module.PROMPT.encode()),
        "image_sha256": reader_module._sha(b"png-old"),
        "raw": '{"rows":[["vintage"]]}',
        "error": None,
        "cost": 0.0021,
    }))
    r = read_png(b"png-old", transport=transport_returning())  # any call fails the test
    assert r.cached and r.rows == (("vintage",),) and r.cost == 0.0


def test_usage_dict_without_cost_records_tokens_and_zero_cost():
    def google_shaped(png):
        return '{"rows":[["ok"]]}', {"prompt_tokens": 800, "completion_tokens": 40}

    r = read_png(b"png-usage-1", transport=google_shaped)
    assert r.rows == (("ok",),) and r.cost == 0.0
    entry = json.loads(reader_module._cache_path(b"png-usage-1").read_text())
    assert entry["cost"] == 0.0
    assert entry["prompt_tokens"] == 800 and entry["completion_tokens"] == 40


def test_usage_dict_with_cost_records_both():
    def openrouter_shaped(png):
        return '{"rows":[["ok"]]}', {"cost": 0.0042, "prompt_tokens": 800,
                                     "completion_tokens": 40}

    r = read_png(b"png-usage-2", transport=openrouter_shaped)
    assert r.cost == pytest.approx(0.0042)
    entry = json.loads(reader_module._cache_path(b"png-usage-2").read_text())
    assert entry["cost"] == pytest.approx(0.0042)
    assert entry["prompt_tokens"] == 800 and entry["completion_tokens"] == 40


def test_legacy_float_transports_record_cost_and_zero_tokens():
    # the pre-#95 hook contract — (text, cost) — still conforms verbatim
    r = read_png(b"png-usage-3", transport=transport_returning('{"rows":[["ok"]]}'))
    assert r.cost == pytest.approx(0.001)
    entry = json.loads(reader_module._cache_path(b"png-usage-3").read_text())
    assert entry["cost"] == pytest.approx(0.001)
    assert entry["prompt_tokens"] == 0 and entry["completion_tokens"] == 0


# --- merged from test_read_table.py ---

@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "READ_CACHE_DIR", str(tmp_path / "reads"))
    monkeypatch.setattr(config, "READER_RETRY_SLEEP_S", 0.0)


def canned(text):
    def _t(png):
        assert png[:8] == b"\x89PNG\r\n\x1a\n", "transport must receive a rendered PNG"
        return text, 0.001

    return _t


TABLE = [
    (72, 100, "Item"), (200, 100, "Amount"),
    (72, 120, "Revenue"), (200, 120, "$1,234"),
    (72, 140, "Expenses"), (200, 140, "$5,678"),
]
FULL_READ = '{"header_rows":1,"rows":[["Item","Amount"],["Revenue","$1,234"],["Expenses","$5,678"]]}'


def table_element(element):
    return element(type="table", bbox=(60, 80, 300, 150), order=0)


def test_grid_rows_kinds_and_claims(make_pdf, element):
    path = make_pdf(TABLE)
    el = table_element(element)
    grid = read_table(path, el, transport=canned(FULL_READ))

    assert grid.element is el
    # header marking is model-first only: the read declared header_rows=1
    assert [r.kind for r in grid.rows] == ["header", "data", "data"]
    assert [c.text for c in grid.rows[0].cells] == ["Item", "Amount"]

    # every cell fully anchored: no misses, provenance page set
    cells = [c for r in grid.rows for c in r.cells]
    assert all(not c.missed and c.page == 1 for c in cells)
    # the element's whole text layer is claimed, each word exactly once
    claimed = [gi for c in cells for gi in c.claims]
    assert sorted(claimed) == sorted({w.index for w in ingest(path).pages[0].words})


def test_words_the_model_missed_land_on_the_unclaimed_channel(make_pdf, element):
    path = make_pdf(TABLE)
    # the reader dropped the "Expenses $5,678" line entirely
    partial = '{"header_rows":1,"rows":[["Item","Amount"],["Revenue","$1,234"]]}'
    grid = read_table(path, table_element(element), transport=canned(partial))

    # rows hold only what the model transcribed (descale ruling)
    assert [r.kind for r in grid.rows] == ["header", "data"]
    assert [c.text for c in grid.unclaimed] == ["Expenses", "$5,678"]
    assert all(c.claims and c.page == 1 for c in grid.unclaimed)


def test_no_financial_lines_means_no_header_rows(make_pdf, element):
    path = make_pdf([(72, 100, "Alpha"), (200, 100, "Beta"), (72, 120, "Gamma"), (200, 120, "Delta")])
    el = element(type="table", bbox=(60, 80, 300, 130), order=0)
    grid = read_table(path, el, transport=canned('{"rows":[["Alpha","Beta"],["Gamma","Delta"]]}'))
    assert [r.kind for r in grid.rows] == ["data", "data"]


def test_lines_are_element_scoped_by_gate_containment(make_pdf, element):
    # a word outside element.bbox belongs to some other element: it must
    # not be claimed here (gate's center-in rule, reused)
    path = make_pdf(TABLE + [(72, 700, "Footer")])
    grid = read_table(path, table_element(element), transport=canned(FULL_READ))
    texts = [c.text for r in grid.rows for c in r.cells]
    assert "Footer" not in texts
    footer = next(w for w in ingest(path).pages[0].words if w.text == "Footer")
    assert footer.index not in {gi for r in grid.rows for c in r.cells for gi in c.claims}


def test_pure_transport_failure_raises_table_unread(make_pdf, element):
    def failing(png):
        raise OSError("connection refused")

    path = make_pdf(TABLE)
    with pytest.raises(TableUnread):
        read_table(path, table_element(element), transport=failing)
