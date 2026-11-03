"""Every tunable in one place: value, unit, and the one-line reason."""

import os
from pathlib import Path

# This package dir. Cache paths resolve against it, never the process CWD.
_SERVICE_ROOT = Path(__file__).resolve().parent

# --- ingest ----------------------------------------------------------------

# A word joins a line when vertical intervals overlap by this fraction of the
# shorter height. A ratio, so the rule is font-size invariant.
LINE_OVERLAP_MIN = 0.5

# A page with fewer text-layer words than this takes the no-text-layer lane:
# proposals pass the gate untouched and the table lane reads blind, low-trust.
TEXT_LAYER_MIN_WORDS = 1

# A word painted again within this distance is the same printed word
# (faux-bold/shadow double-draws). Points; half a thin stroke.
DOUBLE_DRAW_EPSILON_PT = 0.5

# --- coverage gate (gate.py) -----------------------------------------------

# Margin around a snapped element's word hull so renders keep a legibility fringe.
SNAP_MARGIN_PT = 4.0

# An uncovered word joins a cluster hull when within ROW_GAP vertically and
# COL_GAP horizontally: about one line of leading, about a wide word space.
GATE_ROW_GAP_PT = 14.0
GATE_COL_GAP_PT = 40.0

# --- table-region rendering ------------------------------------------------

# 150 DPI is sufficient for the value ledger; pad matches the snap margin so
# the render never clips ink the gate proved is inside.
RENDER_DPI = 150
RENDER_PAD_PT = 4.0

# --- reader transport ------------------------------------------------------

# Any OpenAI-compatible chat-completions endpoint. The model keeps its
# provider prefix; the cache is keyed on that spelling (see reader._cache_model).
READER_MODEL = "google/gemini-3.7-flash"
READER_API_URL = os.environ.get(
    "READER_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
)
# Prefect Secret block holding the key; one pinned block, since bebop is one
# shared service per host. reader._key() reads $READER_KEY_BLOCK first
# ($OPENROUTER_KEY_BLOCK honored as the legacy spelling).
READER_KEY_BLOCK = "gemini-api-key"

# No `reasoning` parameter: the provider clamps output when present.
# Temperature 0 for determinism-in-intent; unparseable responses retry.
READER_MAX_TOKENS = 60000
READER_TEMPERATURE = 0
READER_RETRIES = 3
READER_RETRY_SLEEP_S = 2.0
READER_TIMEOUT_S = 600

# HTTP 429/5xx retry envelope: min(CAP, BASE * 2**k), jittered, behind a
# shared cool-off latch. Six doublings from 2s reach the 60s cap.
READER_THROTTLE_RETRIES = 6
READER_THROTTLE_BASE_S = 2.0
READER_THROTTLE_CAP_S = 60.0

# Reads are network-bound, so width is nearly free until the provider
# throttles. Overridable per host.
READER_PARALLELISM = int(os.environ.get("READER_PARALLELISM", "24"))

# Raw responses, keyed by (model, prompt hash, image hash): the audit trail
# and free replay. _SERVICE_ROOT-anchored.
READ_CACHE_DIR = "cache/reads"

# --- layout model ----------------------------------------------------------

# Raw layout proposals, keyed by (document sha256, layout model version).
LAYOUT_CACHE_DIR = "cache/layout"

# --- composition (compose.py) ----------------------------------------------

# Band merge: the majority of one band's ink must lie inside the other's
# column footprint. Same-layout stacks coincide; different layouts interleave.
FOOTPRINT_ALIGN_MIN = 0.5

# Spread join: this fraction of the smaller side's data rows must y-pair
# within the tolerance across the fold.
SPREAD_Y_TOL_PT = 0.5
SPREAD_MATCH_MIN = 0.5

# Continuation header inheritance: a headerless table adopts the nearest
# earlier same-width table's headers only when both bbox edges agree within
# this tolerance. Measured: true continuations <= 12.3pt, distinct >= 31.5pt.
CONTINUATION_X_TOL_PT = 20.0
