"""
NYTHOS PLUS - adaptive effort runtime for local AI systems (single file, stdlib only).

Principle: do not always make the model think more; let the runtime decide when more compute is
worth it (minimum sufficient effort). Effort levels: E0 DIRECT, E1 FOCUSED, E2 DELIBERATE,
E3 VERIFIED, E4 ESCALATED.

What this is NOT: it does not retrain or modify models, does not read hidden activations or
private neural states, and never stores hidden/private chain-of-thought. It uses only observable
runtime information (visible output, token usage, latency, objective check results).

Native vs emulated effort is always labelled. Native effort control is used only when it was
configured or verified by an explicit probe; otherwise effort is emulated with bounded external
mechanisms (prompt hints, a verification pass, a repair pass). Emulated effort is NOT identical to
changing a model's internal reasoning.

User owns model lifecycle: this program never loads, unloads, downloads or switches models, and
refuses to send inference requests to a model that LM Studio does not report as loaded.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import dataclasses
import datetime
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import socket
import sqlite3
import statistics
import subprocess  # bounded diagnostics only (nvidia-smi / lms / self-test protocol probe); never shell=True
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# =====================================================================================
# CONSTANTS
# =====================================================================================
VERSION = "1.0.0"
APP = "NYTHOS PLUS"
TAGLINE = "Adaptive Effort Runtime"
SCHEMA_VERSION = 1
SERVER_KEY = "nythosplus"
OWNER_ENV_KEY = "NYTHOSPLUS_OWNER"
OWNER_PREFIX = "nythosplus-owned:"
PLUGIN_MARKER = ".nythosplus-owned"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
ID_RE = re.compile(ID_PATTERN)
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TMP_SUFFIX = ".nythosplus-tmp"
NA = "NOT_AVAILABLE"
EFFORT_NAMES = {0: "E0 DIRECT", 1: "E1 FOCUSED", 2: "E2 DELIBERATE", 3: "E3 VERIFIED", 4: "E4 ESCALATED"}
FAMILIES = ("reasoning", "coding", "math", "planning", "instruction", "verification", "consistency",
            "tool_use", "long_context", "vision", "general")
RUN_MODES = ("RAW", "PULSE", "ORACLE", "FIXED_E0", "FIXED_E1", "FIXED_E2", "FIXED_E3", "FIXED_E4")
RES_LEVELS = ("NORMAL", "ELEVATED", "HIGH", "CRITICAL")
LOG = logging.getLogger("nythosplus")


class Hard:
    """Hard ceilings. Config/budgets can never exceed these."""
    MAX_EFFORT = 4
    MAX_CALLS = 12
    MAX_PASSES = 6
    MAX_TOKENS = 65536
    MAX_ELAPSED = 1800.0
    MAX_ESCALATIONS = 4
    MAX_PROMPT = 100_000
    MAX_MCP_PROMPT = 8000
    MAX_OUTPUT_EVAL = 200_000
    MAX_LINE_BYTES = 1_000_000
    MAX_RESULT_CHARS = 24000


def now() -> float:
    return time.time()


def iso(ts: Optional[float]) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else "-"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def is_windows() -> bool:
    return os.name == "nt"


class PlusError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class BudgetStop(Exception):
    """Raised internally when the budget envelope forbids another call (not an error for the user)."""


# =====================================================================================
# PATHS / FILESYSTEM HELPERS
# =====================================================================================
def plus_home() -> Path:
    env = os.environ.get("NYTHOSPLUS_HOME")
    if env:
        return Path(env)
    if is_windows():
        return Path(os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")) / "NythosPlus"
    return Path(os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")) / "NythosPlus"


class Paths:
    def __init__(self, home: Optional[Path] = None):
        self.home = Path(home) if home else plus_home()
        self.db = self.home / "nythosplus.db"
        self.config = self.home / "config.json"
        self.backups = self.home / "backups"
        self.backup_index = self.backups / "index.json"
        self.plugin_dir = self.home / "plugin" / "nythosplus"

    def ensure(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.backups.mkdir(parents=True, exist_ok=True)
        if not is_windows():
            with contextlib.suppress(OSError):
                os.chmod(self.home, 0o700)

    @staticmethod
    def script() -> Path:
        return Path(__file__).resolve()


def atomic_write(path: Path, data: Any) -> None:
    """temp -> write -> flush -> fsync -> os.replace; the target is never left half-written."""
    path = Path(path)
    if isinstance(data, str):
        data = data.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=TMP_SUFFIX, dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    if not is_windows():
        with contextlib.suppress(OSError):
            dfd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)


def clean_stale_tmp(directory: Path, max_age: float = 300.0) -> int:
    n = 0
    with contextlib.suppress(OSError):
        for p in Path(directory).glob("." + "*" + TMP_SUFFIX):
            with contextlib.suppress(OSError):
                if now() - p.stat().st_mtime > max_age:
                    p.unlink()
                    n += 1
    return n


def read_state(paths: Paths) -> Tuple[Dict[str, Any], str]:
    if not paths.config.exists():
        return {}, ""
    try:
        d = json.loads(paths.config.read_text(encoding="utf-8"))
        return (d, "") if isinstance(d, dict) else ({}, "config.json is not a JSON object")
    except (OSError, ValueError) as exc:
        return {}, f"config.json unreadable: {exc}"


def write_state(paths: Paths, state: Dict[str, Any]) -> None:
    state = dict(state)
    state["schema"] = 1
    atomic_write(paths.config, json.dumps(state, indent=2, ensure_ascii=False) + "\n")


# =====================================================================================
# CONFIG + BUDGET
# =====================================================================================
@dataclasses.dataclass
class Budget:
    max_effort: int = 4
    max_total_calls: int = 6
    max_extra_passes: int = 3
    max_tokens: int = 8192
    max_tokens_per_call: int = 2048
    max_reasoning_tokens: Optional[int] = None
    max_elapsed: float = 300.0
    max_escalations: int = 2
    max_verification_cost: float = 2.0

    def clamped(self) -> "Budget":
        b = dataclasses.replace(self)
        b.max_effort = int(clamp(b.max_effort, 0, Hard.MAX_EFFORT))
        b.max_total_calls = int(clamp(b.max_total_calls, 1, Hard.MAX_CALLS))
        b.max_extra_passes = int(clamp(b.max_extra_passes, 0, Hard.MAX_PASSES))
        b.max_tokens = int(clamp(b.max_tokens, 64, Hard.MAX_TOKENS))
        b.max_tokens_per_call = int(clamp(b.max_tokens_per_call, 32, b.max_tokens))
        b.max_elapsed = float(clamp(b.max_elapsed, 5.0, Hard.MAX_ELAPSED))
        b.max_escalations = int(clamp(b.max_escalations, 0, Hard.MAX_ESCALATIONS))
        b.max_verification_cost = float(clamp(b.max_verification_cost, 0.0, 50.0))
        if b.max_reasoning_tokens is not None:
            b.max_reasoning_tokens = int(clamp(b.max_reasoning_tokens, 0, Hard.MAX_TOKENS))
        return b

    @property
    def allowed_calls(self) -> int:
        return min(self.max_total_calls, 1 + self.max_extra_passes)


@dataclasses.dataclass
class Config:
    base_url: str = "http://127.0.0.1:1234"
    http_timeout: float = 120.0
    budget: Budget = dataclasses.field(default_factory=Budget)
    w_tok: float = 1.0            # cost units per 1k tokens
    w_lat: float = 0.1            # cost units per second
    lam: float = 0.02             # quality value of one cost unit (utility = quality - lam*cost - risk)
    w_risk: float = 0.5
    mv_min: float = 0.02          # minimum marginal value (quality per cost unit) to justify a step up
    util_margin: float = 0.005
    enter_thr: float = 0.30       # uncertainty needed to ENTER a higher effort (hysteresis)
    exit_thr: float = 0.15        # uncertainty must fall to this to LEAVE a higher effort
    stop_conf: float = 0.80
    min_evidence: float = 0.50    # outcomes below this evidence strength are not learned from
    cold_k: float = 6.0           # prior strength (pseudo-samples)
    min_samples: int = 8
    learned_min: int = 40
    learning_enabled: bool = True
    explore_eps: float = 0.05
    ucb_c: float = 0.12
    dissent_thr: float = 0.5
    res_thresholds: Tuple[float, float, float] = (70.0, 85.0, 95.0)
    native_efforts: Dict[str, List[str]] = dataclasses.field(default_factory=dict)  # model_id -> values low..high
    warning: str = ""

    @classmethod
    def load(cls, paths: Paths) -> "Config":
        cfg = cls()
        st, err = read_state(paths)
        cfg.warning = err
        s = st.get("settings") if isinstance(st.get("settings"), dict) else {}

        def num(key: str, lo: float, hi: float, integer: bool = False) -> None:
            v = s.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                setattr(cfg, key, int(clamp(v, lo, hi)) if integer else float(clamp(v, lo, hi)))
        for k, lo, hi in (("http_timeout", 1, 600), ("w_tok", 0, 100), ("w_lat", 0, 100), ("lam", 0, 1),
                          ("w_risk", 0, 5), ("mv_min", 0, 10), ("util_margin", 0, 0.2), ("enter_thr", 0.05, 0.95),
                          ("exit_thr", 0.0, 0.9), ("stop_conf", 0.3, 0.999), ("min_evidence", 0.1, 1.0),
                          ("cold_k", 1, 100), ("explore_eps", 0, 0.2), ("ucb_c", 0, 1), ("dissent_thr", 0.1, 1.0)):
            num(k, lo, hi)
        for k, lo, hi in (("min_samples", 2, 1000), ("learned_min", 5, 100000)):
            num(k, lo, hi, True)
        if isinstance(s.get("learning_enabled"), bool):
            cfg.learning_enabled = s["learning_enabled"]
        if isinstance(s.get("base_url"), str):
            with contextlib.suppress(PlusError):
                cfg.base_url = SecurityGuard.local_url(s["base_url"])
        if cfg.exit_thr >= cfg.enter_thr:  # hysteresis requires distinct thresholds
            cfg.exit_thr = max(0.0, cfg.enter_thr - 0.1)
            cfg.warning = (cfg.warning + " exit_thr must be below enter_thr; adjusted").strip()
        b = s.get("budget")
        if isinstance(b, dict):
            for f in dataclasses.fields(Budget):
                v = b.get(f.name)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                    setattr(cfg.budget, f.name, type(getattr(cfg.budget, f.name) or 0)(v) if f.name != "max_reasoning_tokens" else int(v))
        cfg.budget = cfg.budget.clamped()
        ne = s.get("native_efforts")
        if isinstance(ne, dict):
            for m, vals in ne.items():
                if isinstance(m, str) and isinstance(vals, list) and vals and all(isinstance(x, str) and re.fullmatch(r"[a-z]{3,10}", x) for x in vals):
                    cfg.native_efforts[m] = vals[:6]
        return cfg

    def make_budget(self, over: Optional[Dict[str, Any]] = None) -> Budget:
        b = dataclasses.replace(self.budget)
        for k, v in (over or {}).items():
            if hasattr(b, k) and isinstance(v, (int, float)) and not isinstance(v, bool):
                setattr(b, k, v)
        return b.clamped()


# =====================================================================================
# SECURITY GUARD
# =====================================================================================
class SecurityGuard:
    WIN_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.I)

    @staticmethod
    def text(value: Any, name: str, max_len: int, *, required: bool = True, multiline: bool = True) -> str:
        if value is None:
            raise PlusError("invalid_argument", f"{name}: null is not allowed")
        if not isinstance(value, str):
            raise PlusError("invalid_argument", f"{name}: must be a string")
        v = value.strip()
        if required and not v:
            raise PlusError("invalid_argument", f"{name}: must not be empty")
        if len(v) > max_len:
            raise PlusError("too_long", f"{name}: exceeds {max_len} characters")
        if CTRL_RE.search(v) or (not multiline and re.search(r"[\r\n\t]", v)):
            raise PlusError("invalid_argument", f"{name}: contains control characters")
        try:
            v.encode("utf-8")
        except UnicodeEncodeError:
            raise PlusError("invalid_argument", f"{name}: invalid unicode")
        return v

    @staticmethod
    def ident(value: Any, name: str) -> str:
        if not isinstance(value, str) or not ID_RE.fullmatch(value) or ".." in value:
            raise PlusError("invalid_id", f"{name}: invalid identifier")
        return value

    @staticmethod
    def resolve_inside(base: Path, rel: Any) -> Path:
        if not isinstance(rel, str) or not rel or CTRL_RE.search(rel):
            raise PlusError("path_rejected", "invalid path")
        if rel.startswith(("/", "\\", "~")) or re.match(r"^[A-Za-z]:", rel):
            raise PlusError("path_rejected", "absolute paths are not allowed")
        for p in re.split(r"[\\/]+", rel):
            if p in ("..", ".") or SecurityGuard.WIN_RESERVED.match(p):
                raise PlusError("path_rejected", "path traversal or reserved name")
        base_r = Path(base).resolve()
        target = (base_r / rel).resolve()
        if target != base_r and base_r not in target.parents:
            raise PlusError("path_rejected", "path escapes the allowed directory")
        return target

    @staticmethod
    def local_url(url: Any) -> str:
        """Only plain http to loopback is ever allowed (LM Studio local server)."""
        if not isinstance(url, str) or len(url) > 100:
            raise PlusError("url_rejected", "invalid URL")
        u = urllib.parse.urlsplit(url.strip())
        if u.scheme != "http" or u.username or u.password or u.query or u.fragment or u.path not in ("", "/"):
            raise PlusError("url_rejected", "only http://127.0.0.1:<port> style base URLs are allowed")
        host = u.hostname or ""
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise PlusError("url_rejected", "only loopback hosts are allowed (no remote network access)")
        try:
            port = u.port or 80
        except ValueError:
            raise PlusError("url_rejected", "bad port")
        if not 1 <= port <= 65535:
            raise PlusError("url_rejected", "bad port")
        return f"http://{'[::1]' if host == '::1' else host}:{port}"

    @staticmethod
    def check_regex(p: str) -> str:
        if len(p) > 200:
            raise PlusError("invalid_argument", "regex too long")
        if re.search(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)\s*[+*{]", p):
            raise PlusError("invalid_argument", "regex with nested quantifiers rejected")
        try:
            re.compile(p)
        except re.error:
            raise PlusError("invalid_argument", "invalid regex")
        return p

    @staticmethod
    def validate(schema: Dict[str, Any], args: Any) -> Dict[str, Any]:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise PlusError("invalid_argument", "arguments must be an object")
        props = schema.get("properties", {})
        unknown = sorted(set(args) - set(props))
        if unknown:
            raise PlusError("unknown_argument", f"unknown argument(s): {', '.join(map(str, unknown))[:100]}")
        for req in schema.get("required", []):
            if req not in args or args[req] is None:
                raise PlusError("invalid_argument", f"missing required argument: {req}")
        out: Dict[str, Any] = {}
        for key, val in args.items():
            spec, t = props[key], props[key]["type"]
            if val is None:
                raise PlusError("invalid_argument", f"{key}: null is not allowed")
            if t == "string":
                v = SecurityGuard.text(val, key, spec.get("maxLength", 2000), required=spec.get("minLength", 0) > 0)
                if "pattern" in spec and not re.fullmatch(spec["pattern"], v):
                    raise PlusError("invalid_id", f"{key}: invalid format")
                if "enum" in spec and v not in spec["enum"]:
                    raise PlusError("invalid_argument", f"{key}: must be one of {list(spec['enum'])}")
                out[key] = v
            elif t == "integer":
                if isinstance(val, bool) or not isinstance(val, int) or not spec.get("minimum", -10**9) <= val <= spec.get("maximum", 10**9):
                    raise PlusError("out_of_range", f"{key}: must be an integer in range")
                out[key] = val
            elif t == "number":
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) \
                        or not spec.get("minimum", -1e12) <= val <= spec.get("maximum", 1e12):
                    raise PlusError("out_of_range", f"{key}: must be a finite number in range")
                out[key] = float(val)
            elif t == "boolean":
                if not isinstance(val, bool):
                    raise PlusError("invalid_argument", f"{key}: must be a boolean")
                out[key] = val
            elif t == "array":
                if not isinstance(val, list) or len(val) > spec.get("maxItems", 12):
                    raise PlusError("invalid_argument", f"{key}: must be an array (max {spec.get('maxItems', 12)})")
                item = spec.get("items", {"type": "string", "maxLength": 300})
                out[key] = [SecurityGuard.validate({"properties": {"i": item}, "required": ["i"]}, {"i": x})["i"] for x in val]
            else:
                raise PlusError("internal", "unsupported schema type")
        return out


# =====================================================================================
# STORE (SQLite, versioned schema, transactions)
# =====================================================================================
MIGRATIONS: Dict[int, List[str]] = {
    1: [
        """CREATE TABLE IF NOT EXISTS models(
            model_id TEXT PRIMARY KEY, capabilities TEXT NOT NULL DEFAULT '{}',
            native_state TEXT NOT NULL DEFAULT 'NONE', native_values TEXT NOT NULL DEFAULT '[]',
            first_seen REAL NOT NULL, updated_at REAL NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS decisions(
            trace_id TEXT PRIMARY KEY, created_at REAL NOT NULL, model TEXT NOT NULL, family TEXT NOT NULL,
            effort INTEGER NOT NULL, mode TEXT NOT NULL, phase TEXT NOT NULL, policy_confidence REAL NOT NULL,
            kind TEXT NOT NULL, sig TEXT NOT NULL DEFAULT '', pred_q REAL, payload TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN('pending','evaluated','expired')))""",
        """CREATE TABLE IF NOT EXISTS outcomes(
            trace_id TEXT PRIMARY KEY, created_at REAL NOT NULL, bench_id TEXT, task_id TEXT,
            run_mode TEXT NOT NULL, model TEXT NOT NULL, family TEXT NOT NULL, effort INTEGER, final_effort INTEGER,
            kind TEXT, input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER, latency_ms REAL,
            ttft_ms REAL, tps REAL, verification TEXT, quality REAL, evidence REAL, success INTEGER,
            repair INTEGER NOT NULL DEFAULT 0, escalations INTEGER NOT NULL DEFAULT 0, calls INTEGER NOT NULL DEFAULT 1,
            cost REAL, resource_state TEXT, policy_confidence REAL, steps TEXT)""",
        """CREATE TABLE IF NOT EXISTS attempts(
            attempt_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL REFERENCES outcomes(trace_id) ON DELETE CASCADE,
            idx INTEGER NOT NULL, model TEXT NOT NULL, family TEXT NOT NULL, kind TEXT NOT NULL, effort INTEGER NOT NULL,
            clean INTEGER NOT NULL, quality REAL, evidence REAL, pred_q REAL, sig TEXT NOT NULL DEFAULT '',
            input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER, latency_ms REAL, ttft_ms REAL,
            calls INTEGER NOT NULL DEFAULT 1, verifier TEXT, created_at REAL NOT NULL)""",
        "CREATE INDEX IF NOT EXISTS idx_att_prof ON attempts(model, family, kind, effort)",
        "CREATE INDEX IF NOT EXISTS idx_out_model ON outcomes(model, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_out_bench ON outcomes(bench_id)",
        """CREATE TABLE IF NOT EXISTS effort_state(key TEXT PRIMARY KEY, effort INTEGER NOT NULL, updated_at REAL NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS inflight(slot TEXT PRIMARY KEY, trace_id TEXT NOT NULL, started REAL NOT NULL, expires REAL NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS events(event_id TEXT PRIMARY KEY, request_id TEXT, kind TEXT NOT NULL,
            ok INTEGER NOT NULL, created_at REAL NOT NULL, detail TEXT NOT NULL DEFAULT '')""",
    ],
}


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self.path), timeout=3.0, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        c.execute("PRAGMA journal_mode = WAL")
        c.execute("PRAGMA synchronous = FULL")
        c.execute("PRAGMA busy_timeout = 3000")
        return c

    @contextlib.contextmanager
    def tx(self):
        with self._lock:
            c = self.connect()
            try:
                c.execute("BEGIN IMMEDIATE")
                yield c
                c.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    c.execute("ROLLBACK")
                raise
            finally:
                c.close()

    @contextlib.contextmanager
    def read(self):
        c = self.connect()
        try:
            yield c
        finally:
            c.close()

    def init_schema(self, migrations: Optional[Dict[int, List[str]]] = None) -> None:
        migrations = migrations if migrations is not None else MIGRATIONS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.tx() as c:
                c.execute("CREATE TABLE IF NOT EXISTS schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                row = c.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
                cur = int(row["value"]) if row else 0
                if cur > max(migrations):
                    raise PlusError("schema_too_new", "database schema is newer than this build; refusing to touch it")
                for v in sorted(migrations):
                    if v > cur:
                        for stmt in migrations[v]:
                            c.execute(stmt)
                        c.execute("INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version',?)", (str(v),))
        except sqlite3.DatabaseError as exc:
            raise PlusError("db_error", f"database unusable ({str(exc)[:80]}); it was NOT modified or deleted")

    def schema_version(self) -> int:
        with self.read() as c:
            r = c.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
            return int(r["value"]) if r else 0

    def integrity(self) -> Tuple[bool, str]:
        try:
            with self.read() as c:
                msgs = [r[0] for r in c.execute("PRAGMA integrity_check")]
                if msgs != ["ok"]:
                    return False, "; ".join(msgs)[:200]
                if c.execute("PRAGMA foreign_key_check").fetchall():
                    return False, "foreign key violations"
                return True, "ok"
        except sqlite3.DatabaseError as exc:
            return False, str(exc)[:200]

    def journal_mode(self) -> str:
        with self.read() as c:
            return str(c.execute("PRAGMA journal_mode").fetchone()[0])

    # ---- single-flight (cross-process, atomic)
    def acquire(self, slot: str, trace_id: str, ttl: float) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM inflight WHERE expires<?", (now(),))
            try:
                c.execute("INSERT INTO inflight(slot,trace_id,started,expires) VALUES(?,?,?,?)", (slot, trace_id, now(), now() + ttl))
            except sqlite3.IntegrityError:
                raise PlusError("busy", "another adaptive-effort run is already in flight for this model (single-flight)")

    def release(self, slot: str, trace_id: str) -> None:
        with contextlib.suppress(sqlite3.Error):
            with self.tx() as c:
                c.execute("DELETE FROM inflight WHERE slot=? AND trace_id=?", (slot, trace_id))

    def event(self, request_id: str, kind: str, ok: bool, detail: str = "") -> None:
        with contextlib.suppress(Exception):
            with self.tx() as c:
                c.execute("INSERT INTO events(event_id,request_id,kind,ok,created_at,detail) VALUES(?,?,?,?,?,?)",
                          (new_id("e"), request_id, kind[:60], int(ok), now(), detail[:120]))
                c.execute("DELETE FROM events WHERE rowid IN (SELECT rowid FROM events ORDER BY created_at LIMIT MAX(0,(SELECT COUNT(*) FROM events)-5000))")


class RecursionGuard:
    """Anti-recursion: a pulse run can never start from inside another pulse run in this thread."""
    _local = threading.local()

    def __enter__(self):
        if getattr(self._local, "depth", 0) > 0:
            raise PlusError("recursion_blocked", "adaptive-effort run requested from inside another run; blocked")
        self._local.depth = 1
        return self

    def __exit__(self, *a):
        self._local.depth = 0


# =====================================================================================
# MODEL PROFILE
# =====================================================================================
@dataclasses.dataclass
class ModelProfile:
    model_id: str
    capabilities: Dict[str, Any] = dataclasses.field(default_factory=dict)
    native_state: str = "NONE"          # NONE | HINT | CONFIG | VERIFIED | INEFFECTIVE | UNSUPPORTED
    native_values: List[str] = dataclasses.field(default_factory=list)
    tps_mean: Optional[float] = None
    latency_mean: Optional[float] = None
    ttft_mean: Optional[float] = None
    in_tokens_mean: Optional[float] = None
    out_tokens_mean: Optional[float] = None
    first_pass_success: Optional[float] = None
    verification_success: Optional[float] = None
    repair_rate: Optional[float] = None
    escalation_rate: Optional[float] = None
    n_runs: int = 0

    def kind(self) -> str:
        return "native" if self.native_state in ("CONFIG", "VERIFIED") and self.native_values else "emulated"

    def native_value(self, effort: int) -> Optional[str]:
        if self.kind() != "native":
            return None
        v = self.native_values
        if effort <= 1:
            return v[0]
        if effort == 2:
            return v[len(v) // 2]
        return v[-1]

    def mode_label(self, effort: int, raw: bool = False) -> str:
        if raw:
            return "RAW"
        if self.kind() == "native":
            return "NATIVE" if effort in (0, 2) else "HYBRID"
        return "EMULATED"


class ProfileStore:
    def __init__(self, store: Store, cfg: Config):
        self.store, self.cfg = store, cfg

    def get(self, model_id: str) -> ModelProfile:
        with self.store.read() as c:
            r = c.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
            p = ModelProfile(model_id)
            if r:
                p.capabilities = json.loads(r["capabilities"] or "{}")
                p.native_state, p.native_values = r["native_state"], json.loads(r["native_values"] or "[]")
            if model_id in self.cfg.native_efforts:
                p.native_state, p.native_values = "CONFIG", list(self.cfg.native_efforts[model_id])
            elif not r and re.search(r"gpt-oss", model_id, re.I):
                p.native_state = "HINT"  # id suggests native effort control; NOT used until verified via `models --probe`
            a = c.execute("SELECT AVG(tps) tps, AVG(latency_ms) lat, AVG(ttft_ms) ttft, AVG(input_tokens) it, AVG(output_tokens) ot, COUNT(*) n "
                          "FROM outcomes WHERE model=?", (model_id,)).fetchone()
            p.n_runs = a["n"] or 0
            p.tps_mean, p.latency_mean, p.ttft_mean, p.in_tokens_mean, p.out_tokens_mean = a["tps"], a["lat"], a["ttft"], a["it"], a["ot"]
            if p.n_runs:
                s = c.execute("SELECT AVG(CASE WHEN quality>=0.999 AND evidence>=? THEN 1.0 ELSE 0.0 END) fps, "
                              "AVG(repair) rep, AVG(CASE WHEN escalations>0 THEN 1.0 ELSE 0.0 END) esc FROM outcomes WHERE model=?",
                              (self.cfg.min_evidence, model_id)).fetchone()
                p.first_pass_success, p.repair_rate, p.escalation_rate = s["fps"], s["rep"], s["esc"]
                v = c.execute("SELECT COUNT(*) n, AVG(CASE WHEN o.quality>(SELECT a2.quality FROM attempts a2 WHERE a2.trace_id=o.trace_id "
                              "ORDER BY a2.idx LIMIT 1) THEN 1.0 ELSE 0.0 END) y FROM outcomes o WHERE o.model=? AND o.verification IS NOT NULL "
                              "AND o.verification!='NONE'", (model_id,)).fetchone()
                p.verification_success = v["y"] if v["n"] else None
        return p

    def upsert(self, model_id: str, caps: Optional[Dict[str, Any]] = None, native_state: Optional[str] = None,
               native_values: Optional[List[str]] = None) -> None:
        with self.store.tx() as c:
            r = c.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
            ts = now()
            if r is None:
                c.execute("INSERT INTO models(model_id,capabilities,native_state,native_values,first_seen,updated_at) VALUES(?,?,?,?,?,?)",
                          (model_id, json.dumps(caps or {}), native_state or "NONE", json.dumps(native_values or []), ts, ts))
            else:
                c.execute("UPDATE models SET capabilities=?, native_state=?, native_values=?, updated_at=? WHERE model_id=?",
                          (json.dumps(caps if caps is not None else json.loads(r["capabilities"])),
                           native_state or r["native_state"],
                           json.dumps(native_values if native_values is not None else json.loads(r["native_values"])), ts, model_id))

    def effort_stats(self, model: str, family: str, kind: str) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        with self.store.read() as c:
            for r in c.execute(
                    "SELECT effort, SUM(CASE WHEN quality IS NOT NULL AND evidence>=? THEN 1 ELSE 0 END) nq, "
                    "AVG(CASE WHEN quality IS NOT NULL AND evidence>=? THEN quality END) q, "
                    "SUM(CASE WHEN output_tokens IS NOT NULL THEN 1 ELSE 0 END) nt, AVG(output_tokens) ot, AVG(input_tokens) it, "
                    "AVG(latency_ms) lat, AVG(calls) calls FROM attempts WHERE model=? AND family=? AND kind=? AND clean=1 GROUP BY effort",
                    (self.cfg.min_evidence, self.cfg.min_evidence, model, family, kind)):
                out[r["effort"]] = {"nq": r["nq"] or 0, "q": r["q"], "nt": r["nt"] or 0, "ot": r["ot"], "it": r["it"],
                                    "lat": r["lat"], "calls": r["calls"]}
        return out


# =====================================================================================
# FAST PROBE (cheap request features; no model call)
# =====================================================================================
_RE = {
    "constraint": re.compile(r"\b(must|should|exactly|at most|at least|no more than|only|without|do not|don't|never|ensure|required|limit)\b", re.I),
    "ops": re.compile(r"\b(write|create|build|implement|explain|list|compare|summari[sz]e|translate|calculate|solve|design|generate|analy[sz]e|fix|refactor|review|plan|prove|derive|convert|extract|classify)\b", re.I),
    "multi": re.compile(r"\b(first|second|third|then|next|after that|finally|step[- ]by[- ]step|steps?)\b|^\s*\d+[.)]\s", re.I | re.M),
    "ambig": re.compile(r"\b(maybe|perhaps|somehow|something|etc|and so on|whatever|best way|some kind of|or so)\b", re.I),
    "verify": re.compile(r"\b(verify|double-check|check that|prove|validate|make sure|ensure|test)\b", re.I),
    "code": re.compile(r"[{};]|=>|\bdef\b|\bfunction\b|^\s{4,}\S|==|\(\)", re.M),
}
_FAMILY_RE = {
    "coding": re.compile(r"```|\b(function|class|def|python|javascript|typescript|bug|compile|implement|refactor|regex|sql|api endpoint|unit test|code)\b", re.I),
    "math": re.compile(r"\b(calculate|solve|equation|integral|derivative|probability|sum of|percent|average|how many|compute)\b|\d\s*[-+*/^]\s*\d", re.I),
    "reasoning": re.compile(r"\b(why|deduce|logic|puzzle|riddle|infer|taller|shorter|older|younger|therefore|all .* are)\b", re.I),
    "planning": re.compile(r"\b(plan|schedule|roadmap|steps to|strategy|itinerary|timeline|milestones?)\b", re.I),
    "instruction": re.compile(r"\b(exactly|format|bullet|words|in json|json only|do not include|must start with)\b", re.I),
    "verification": re.compile(r"\b(verify|find the bug|is this correct|is the following|validate|audit|review this|fact-check)\b", re.I),
    "consistency": re.compile(r"\b(consistent|contradict|reconcile|compare the two|same answer)\b", re.I),
    "tool_use": re.compile(r"\b(tool call|function call|json schema|arguments|call the|invoke)\b", re.I),
}


@dataclasses.dataclass
class RequestFeatures:
    chars: int = 0
    approx_tokens: int = 0
    constraints: int = 0
    operations: int = 0
    questions: int = 0
    code_density: float = 0.0
    numeric_density: float = 0.0
    ambiguity: int = 0
    multistep: int = 0
    verify_req: int = 0
    has_image: bool = False
    history_failures: int = 0
    novelty: float = 1.0
    family: str = "general"
    complexity: float = 0.0
    sig: str = ""

    def public(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        for k in ("code_density", "numeric_density", "novelty", "complexity"):
            d[k] = round(d[k], 3)
        return d


def probe_prompt(prompt: str, has_image: bool = False) -> RequestFeatures:
    f = RequestFeatures(chars=len(prompt), has_image=has_image)
    f.approx_tokens = max(1, int(len(prompt) / 4))
    f.constraints = len(_RE["constraint"].findall(prompt))
    f.operations = len(set(m.lower() for m in _RE["ops"].findall(prompt)))
    f.questions = prompt.count("?")
    lines = [ln for ln in prompt.splitlines() if ln.strip()] or [prompt]
    code_lines = sum(1 for ln in lines if _RE["code"].search(ln))
    fenced = len(re.findall(r"```", prompt)) // 2
    f.code_density = min(1.0, code_lines / len(lines) + 0.3 * fenced)
    f.numeric_density = min(1.0, len(re.findall(r"\d+(?:\.\d+)?", prompt)) / max(1, len(prompt.split())))
    f.ambiguity = len(_RE["ambig"].findall(prompt))
    f.multistep = len(_RE["multi"].findall(prompt))
    f.verify_req = len(_RE["verify"].findall(prompt))
    scores = {fam: len(rx.findall(prompt)) for fam, rx in _FAMILY_RE.items()}
    scores["coding"] += int(f.code_density * 4)
    scores["math"] += int(f.numeric_density * 6)
    scores["instruction"] += min(3, f.constraints // 2)
    fam, best = "general", 0.99
    for name in ("coding", "math", "verification", "tool_use", "planning", "consistency", "reasoning", "instruction"):
        if scores[name] > best:
            fam, best = name, scores[name]
    if f.approx_tokens > 3000:
        fam = "long_context"
    if has_image:
        fam = "vision"
    f.family = fam
    f.complexity = clamp(0.18 * min(1, f.constraints / 6) + 0.15 * min(1, f.operations / 5) + 0.15 * min(1, f.multistep / 3)
                         + 0.12 * min(1, f.ambiguity / 3) + 0.15 * min(1, math.log10(f.approx_tokens + 1) / 4)
                         + 0.12 * f.code_density + 0.08 * f.numeric_density + 0.05 * min(1, f.verify_req / 2), 0.0, 1.0)
    f.sig = f"{f.family}|c{int(f.complexity * 4)}|t{min(4, int(math.log10(f.approx_tokens + 1)))}"
    return f


# =====================================================================================
# RESOURCE GUARD (observe only; never kills, unloads or tunes anything)
# =====================================================================================
@dataclasses.dataclass
class ResourceSnapshot:
    cpu: Optional[float]
    mem: Optional[float]
    gpu: Optional[float]
    vram: Optional[float]
    state: str
    observed: bool

    def public(self) -> Dict[str, Any]:
        f = lambda v: NA if v is None else round(v, 1)
        return {"state": self.state if self.observed else NA, "cpu_pct": f(self.cpu), "mem_pct": f(self.mem),
                "gpu_pct": f(self.gpu), "vram_pct": f(self.vram)}


class ResourceGuard:
    def __init__(self, sampler: Optional[Callable[[], Dict[str, Optional[float]]]] = None, thresholds=(70.0, 85.0, 95.0), ttl: float = 2.0):
        self.sampler, self.thr, self.ttl = sampler, thresholds, ttl
        self._cache: Optional[Tuple[float, ResourceSnapshot]] = None
        self._gpu: Optional[Tuple[float, Tuple[Optional[float], Optional[float]]]] = None
        self._win_prev: Optional[Tuple[int, int, int]] = None

    def _gpu_sample(self) -> Tuple[Optional[float], Optional[float]]:
        if self._gpu and now() - self._gpu[0] < 10:
            return self._gpu[1]
        val: Tuple[Optional[float], Optional[float]] = (None, None)
        exe = shutil.which("nvidia-smi")
        if exe:
            with contextlib.suppress(Exception):
                p = subprocess.run([exe, "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=3)
                if p.returncode == 0 and p.stdout.strip():
                    u, mu, mt = [float(x) for x in p.stdout.splitlines()[0].split(",")]
                    val = (u, 100.0 * mu / mt if mt else None)
        self._gpu = (now(), val)
        return val

    def _default(self) -> Dict[str, Optional[float]]:
        cpu = mem = None
        if is_windows():
            with contextlib.suppress(Exception):
                import ctypes

                class MS(ctypes.Structure):
                    _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                                ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong),
                                ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
                ms = MS()
                ms.l = ctypes.sizeof(MS)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):  # type: ignore[attr-defined]
                    mem = float(ms.load)

                class FT(ctypes.Structure):
                    _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]
                i, k, u = FT(), FT(), FT()
                if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u)):  # type: ignore[attr-defined]
                    tv = lambda x: (x.hi << 32) | x.lo
                    cur = (tv(i), tv(k), tv(u))
                    if self._win_prev:
                        di, dk, du = (cur[n] - self._win_prev[n] for n in range(3))
                        tot = dk + du
                        cpu = 100.0 * (tot - di) / tot if tot > 0 else None
                    self._win_prev = cur
        else:
            with contextlib.suppress(Exception):
                cpu = min(100.0, 100.0 * os.getloadavg()[0] / (os.cpu_count() or 1))
            with contextlib.suppress(Exception):
                info = {}
                for ln in Path("/proc/meminfo").read_text().splitlines():
                    k, v = ln.split(":", 1)
                    info[k] = float(v.split()[0])
                mem = 100.0 * (1 - info["MemAvailable"] / info["MemTotal"])
        gpu, vram = self._gpu_sample()
        return {"cpu": cpu, "mem": mem, "gpu": gpu, "vram": vram}

    def classify(self, vals: Dict[str, Optional[float]]) -> Tuple[str, bool]:
        known = [v for v in vals.values() if v is not None]
        if not known:
            return "NORMAL", False
        m = max(known)
        lvl = 0 + (m >= self.thr[0]) + (m >= self.thr[1]) + (m >= self.thr[2])
        return RES_LEVELS[lvl], True

    def sample(self, force: bool = False) -> ResourceSnapshot:
        if self._cache and not force and now() - self._cache[0] < self.ttl:
            return self._cache[1]
        try:
            vals = (self.sampler or self._default)()
        except Exception:
            vals = {}
        st, obs = self.classify({k: vals.get(k) for k in ("cpu", "mem", "gpu", "vram")})
        snap = ResourceSnapshot(vals.get("cpu"), vals.get("mem"), vals.get("gpu"), vals.get("vram"), st, obs)
        self._cache = (now(), snap)
        return snap

    @staticmethod
    def limits(state: str) -> Dict[str, Any]:
        """(effort cap, escalation cap, cost factor, allow new expensive work)."""
        return {"NORMAL": {"cap": 4, "esc": 4, "cost": 1.0, "allow": True},
                "ELEVATED": {"cap": 4, "esc": 2, "cost": 1.5, "allow": True},
                "HIGH": {"cap": 2, "esc": 1, "cost": 2.5, "allow": True},
                "CRITICAL": {"cap": 0, "esc": 0, "cost": 4.0, "allow": False}}[state]


# =====================================================================================
# OUTCOME ASSESSMENT / DISSENT / VERIFICATION ECONOMICS
# =====================================================================================
def parse_check(s: str) -> Dict[str, Any]:
    """Compact check syntax: contains:x not_contains:x equals:x number:403 regex:p json json_keys:a,b json_len:k=3
    python defines:name max_words:N min_words:N words:N bullets:N"""
    s = SecurityGuard.text(s, "check", 300)
    kind, _, arg = s.partition(":")
    kind = kind.strip().lower()
    if kind in ("json", "python"):
        return {"type": kind}
    if not arg and kind not in ("json", "python"):
        raise PlusError("invalid_argument", f"check '{kind}' needs an argument")
    if kind in ("contains", "not_contains", "equals", "defines"):
        return {"type": kind, "value": arg}
    if kind == "regex":
        return {"type": "regex", "value": SecurityGuard.check_regex(arg)}
    if kind == "number":
        try:
            return {"type": "number", "value": float(arg)}
        except ValueError:
            raise PlusError("invalid_argument", "number check needs a numeric argument")
    if kind == "json_keys":
        return {"type": "json_keys", "value": [k.strip() for k in arg.split(",") if k.strip()][:20]}
    if kind == "json_len":
        k, _, n = arg.partition("=")
        if not n.isdigit():
            raise PlusError("invalid_argument", "json_len needs key=N")
        return {"type": "json_len", "key": k.strip(), "value": int(n)}
    if kind in ("max_words", "min_words", "words", "bullets"):
        if not arg.strip().isdigit():
            raise PlusError("invalid_argument", f"{kind} needs an integer")
        return {"type": kind, "value": int(arg)}
    raise PlusError("invalid_argument", f"unknown check type '{kind[:20]}'")


def _norm(t: str) -> str:
    t = t.strip().strip("`*_\"' \n").lower()
    return re.sub(r"[.!\s]+$", "", t)


def _words(t: str) -> int:
    return len(re.findall(r"[\w'-]+", t, re.UNICODE))


def _last_number(t: str) -> Optional[float]:
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    m = re.findall(r"-?\d+(?:\.\d+)?", t)
    return float(m[-1]) if m else None


def _json_in(t: str) -> Any:
    t = t.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    for cand in (t, t[t.find("{"):t.rfind("}") + 1] if "{" in t else "", t[t.find("["):t.rfind("]") + 1] if "[" in t else ""):
        if cand:
            with contextlib.suppress(ValueError):
                return json.loads(cand)
    raise ValueError("no json")


def _py_code(t: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", t, re.S)
    return m.group(1) if m else t


@dataclasses.dataclass
class CheckResult:
    name: str
    passed: bool
    weight: float
    trust: float
    detail: str = ""


@dataclasses.dataclass
class Outcome:
    pass_fraction: float = 0.0
    quality: Optional[float] = None
    evidence_strength: float = 0.0
    system_confidence: float = 0.0
    model_confidence: Optional[float] = None
    results: List[CheckResult] = dataclasses.field(default_factory=list)
    violations: List[str] = dataclasses.field(default_factory=list)
    issues: List[str] = dataclasses.field(default_factory=list)
    verifier: Optional[str] = None
    unscored: bool = True

    def public(self) -> Dict[str, Any]:
        return {"quality": NA if self.quality is None else round(self.quality, 3), "pass_fraction": round(self.pass_fraction, 3),
                "evidence_strength": round(self.evidence_strength, 3), "system_confidence": round(self.system_confidence, 3),
                "model_confidence": NA if self.model_confidence is None else round(self.model_confidence, 3),
                "violations": self.violations[:8], "issues": self.issues, "verifier": self.verifier or "NONE", "scored": not self.unscored}


class OutcomeEvaluator:
    """Objective, task-specific checks only. Model self-confidence and self-reports are never proof. Nothing is executed."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def auto_checks(prompt: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        p = prompt.lower()
        m = re.search(r"exactly (\d+) words", p)
        if m:
            out.append({"type": "words", "value": int(m.group(1)), "source": "auto"})
        m = re.search(r"(?:at most|no more than|maximum of|under) (\d+) words", p)
        if m:
            out.append({"type": "max_words", "value": int(m.group(1)), "source": "auto"})
        m = re.search(r"at least (\d+) words", p)
        if m:
            out.append({"type": "min_words", "value": int(m.group(1)), "source": "auto"})
        m = re.search(r"exactly (\d+) (?:bullet|bullets|bullet points)", p)
        if m:
            out.append({"type": "bullets", "value": int(m.group(1)), "source": "auto"})
        if re.search(r"\b(json only|valid json|return json|respond in json|answer in json)\b", p):
            out.append({"type": "json", "source": "auto"})
        for m in re.finditer(r"(?:do not (?:include|mention|use) the word|without (?:using )?the word) ['\"]?(\w+)", p):
            out.append({"type": "not_contains", "value": m.group(1), "source": "auto"})
        return out[:8]

    def run_check(self, spec: Dict[str, Any], text: str) -> Tuple[bool, str]:
        t, v = spec["type"], spec.get("value")
        try:
            if t == "contains":
                return str(v).lower() in text.lower(), f"contains {str(v)[:30]!r}"
            if t == "not_contains":
                return str(v).lower() not in text.lower(), f"absent {str(v)[:30]!r}"
            if t == "equals":
                return _norm(text) == _norm(str(v)), f"equals {str(v)[:30]!r}"
            if t == "number":
                n = _last_number(text)
                return (n is not None and abs(n - float(v)) <= 1e-6 * max(1.0, abs(float(v)))), f"number {v}, got {NA if n is None else n}"
            if t == "regex":
                return re.search(SecurityGuard.check_regex(str(v)), text[:50_000]) is not None, "regex"
            if t == "json":
                _json_in(text)
                return True, "valid json"
            if t == "json_keys":
                d = _json_in(text)
                return isinstance(d, dict) and all(k in d for k in v), f"keys {v}"
            if t == "json_len":
                d = _json_in(text)
                x = d.get(spec["key"]) if isinstance(d, dict) else None
                return isinstance(x, list) and len(x) == int(v), f"{spec['key']} length {int(v)}"
            if t == "python":
                ast.parse(_py_code(text))
                return True, "python syntax"
            if t == "defines":
                tree = ast.parse(_py_code(text))
                return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == v for n in ast.walk(tree)), f"defines {v}"
            if t == "words":
                n = _words(text)
                return n == int(v), f"exactly {int(v)} words, got {n}"
            if t == "max_words":
                n = _words(text)
                return n <= int(v), f"at most {int(v)} words, got {n}"
            if t == "min_words":
                n = _words(text)
                return n >= int(v), f"at least {int(v)} words, got {n}"
            if t == "bullets":
                n = len([ln for ln in text.splitlines() if re.match(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S", ln)])
                return n == int(v), f"exactly {int(v)} bullets, got {n}"
        except (ValueError, SyntaxError, TypeError, RecursionError, KeyError, AttributeError):
            return False, f"{t} failed to evaluate"
        return False, f"unknown check {t}"

    def evaluate(self, text: str, finish_reason: Optional[str], checks: List[Dict[str, Any]], prior_q: float = 0.5,
                 verifier: Optional[str] = None) -> Outcome:
        text = text[:Hard.MAX_OUTPUT_EVAL]
        o = Outcome(verifier=verifier)
        for spec in checks:
            ok, detail = self.run_check(spec, text)
            trust = 1.0 if spec.get("source", "user") == "user" else 0.6
            o.results.append(CheckResult(spec["type"], ok, float(spec.get("weight", 1.0)), trust, detail))
            if not ok:
                o.violations.append(detail)
        if not text.strip():
            o.issues.append("empty_output")
        if finish_reason == "length":
            o.issues.append("truncated")
        toks = re.findall(r"\w+", text.lower())
        if len(toks) >= 40:
            grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
            if len(set(grams)) / len(grams) < 0.4:
                o.issues.append("repetition_loop")
        wsum = sum(r.weight * r.trust for r in o.results)
        o.pass_fraction = (sum(r.weight * r.trust for r in o.results if r.passed) / wsum) if wsum else 0.0
        sanity_trust = 0.15 + (0.125 if verifier in ("PASS", "FAIL") else 0.0)
        o.evidence_strength = 1.0 - math.exp(-(wsum + sanity_trust) / 1.5)
        if wsum > 0:
            q = o.pass_fraction
            if "empty_output" in o.issues:
                q = 0.0
            if "truncated" in o.issues:
                q *= 0.5
            if "repetition_loop" in o.issues:
                q *= 0.6
            o.quality = q
        o.unscored = o.quality is None or o.evidence_strength < self.cfg.min_evidence
        if wsum == 0 and "empty_output" in o.issues:
            o.quality, o.pass_fraction = 0.0, 0.0
        base = o.pass_fraction if wsum else prior_q * 0.6
        o.system_confidence = clamp(o.evidence_strength * base + (1 - o.evidence_strength) * prior_q * 0.6, 0.0, 1.0)
        if o.issues and wsum == 0:
            o.system_confidence = min(o.system_confidence, 0.3)
        if "truncated" in o.issues or "empty_output" in o.issues:
            o.system_confidence = min(o.system_confidence, 0.4)
        m = re.search(r"confidence\W{0,3}(\d+(?:\.\d+)?)\s*(%?)", text, re.I)
        if m:
            x = float(m.group(1))
            o.model_confidence = clamp(x / 100.0 if (m.group(2) or x > 1.0) else x, 0.0, 1.0)
        return o


@dataclasses.dataclass
class Claim:
    source: str
    text: str
    confidence: Optional[float]
    evidence: float
    verification: Optional[str]
    contradiction: bool


class DissentTracker:
    """Evidence-weighted disagreement (never a majority vote)."""

    def __init__(self):
        self.claims: List[Claim] = []

    def add(self, source: str, text: str, confidence: Optional[float], evidence: float, verification: Optional[str], contradiction: bool) -> None:
        self.claims.append(Claim(source, text, confidence, evidence, verification, contradiction))

    @staticmethod
    def _key(t: str) -> str:
        n = _last_number(t)
        return f"num:{n}" if n is not None and len(t) < 80 else " ".join(re.findall(r"\w+", t.lower())[:12])

    def score(self) -> float:
        if not self.claims:
            return 0.0
        weights: Dict[str, float] = {}
        for c in self.claims:
            weights[self._key(c.text)] = weights.get(self._key(c.text), 0.0) + max(0.05, c.evidence)
        tot = sum(weights.values())
        disagree = 1.0 - max(weights.values()) / tot if len(self.claims) > 1 else 0.0
        contra = max((0.5 for c in self.claims if c.contradiction), default=0.0)
        return clamp(disagree + contra, 0.0, 1.0)


def verification_value(system_conf: float, repair_rate: Optional[float], verify_cost: float, cfg: Config, remaining_cost: float) -> Dict[str, Any]:
    """Verify only when expected value (chance an undetected error is found and fixed) beats its cost."""
    rr = 0.5 if repair_rate is None else clamp(repair_rate, 0.05, 1.0)
    gain = (1.0 - system_conf) * rr
    cost = cfg.lam * verify_cost
    return {"expected_gain": round(gain, 4), "cost_equiv": round(cost, 4), "justified": gain > cost and verify_cost <= remaining_cost,
            "repair_rate_used": round(rr, 3)}


# =====================================================================================
# EFFORT POLICY (decision core)
# =====================================================================================
@dataclasses.dataclass
class Curve:
    q: List[float]
    q_src: List[str]
    n: List[int]
    tokens: List[float]
    latency: List[float]
    cost: List[float]
    calls: List[int]
    sens: float
    phase: str
    confidence: float
    kind: str
    assumed_tps: bool


@dataclasses.dataclass
class EffortDecision:
    trace_id: str
    model: str
    family: str
    effort: int
    mode: str
    native_value: Optional[str]
    phase: str
    confidence: float
    sensitivity: float
    expected_quality: float
    expected_incremental_cost: float
    marginal_value: float
    q: List[float]
    cost: List[float]
    utility: List[float]
    marginal: List[float]
    cap: int
    resource_state: str
    reasons: List[str]
    explored: bool = False
    kind: str = "emulated"
    sig: str = ""

    def public(self) -> Dict[str, Any]:
        r = lambda xs: [round(x, 4) for x in xs]
        return {"trace_id": self.trace_id, "model": self.model, "task_family": self.family, "effort": EFFORT_NAMES[self.effort],
                "mode": self.mode, "native_value": self.native_value or NA, "policy_phase": self.phase,
                "effort_confidence": round(self.confidence, 3), "effort_sensitivity": round(self.sensitivity, 3),
                "expected_quality": round(self.expected_quality, 3), "expected_incremental_cost": round(self.expected_incremental_cost, 4),
                "marginal_value": round(self.marginal_value, 4), "quality_curve": r(self.q), "cost_curve": r(self.cost),
                "utility_curve": r(self.utility), "marginal_curve": r(self.marginal), "effort_cap": self.cap,
                "resource_state": self.resource_state, "explored": self.explored, "reasons": self.reasons}


class EffortPolicy:
    GAIN_FRAC = (0.0, 0.40, 0.68, 0.82, 0.88)   # share of headroom recovered at each level (diminishing returns)
    TOK_MULT_NATIVE = (1.0, 1.25, 2.2, 3.6, 5.5)
    TOK_MULT_EMUL = (1.0, 1.35, 2.0, 3.2, 5.0)
    CALLS = (1, 1, 1, 2, 3)
    ASSUMED_TPS = 25.0     # cold-start assumption only; replaced by observed throughput as soon as it exists

    def __init__(self, cfg: Config, profiles: ProfileStore):
        self.cfg, self.profiles = cfg, profiles

    def calibration_bias(self, model: str, family: str, kind: str) -> Tuple[float, int]:
        """CalibrationEngine: mean(observed - predicted) for this model/family; shifts the prior curve."""
        with self.profiles.store.read() as c:
            r = c.execute("SELECT COUNT(*) n, AVG(quality - pred_q) b FROM attempts WHERE model=? AND family=? AND kind=? AND clean=1 "
                          "AND quality IS NOT NULL AND pred_q IS NOT NULL AND evidence>=?", (model, family, kind, self.cfg.min_evidence)).fetchone()
        n = r["n"] or 0
        return (clamp(r["b"], -0.15, 0.15) if n >= self.cfg.min_samples and r["b"] is not None else 0.0), n

    def curve(self, f: RequestFeatures, prof: ModelProfile, objective: bool, res_cost: float = 1.0) -> Curve:
        cfg, kind = self.cfg, prof.kind()
        stats = self.profiles.effort_stats(prof.model_id, f.family, kind)
        bias, _ = self.calibration_bias(prof.model_id, f.family, kind)
        n = [stats.get(e, {}).get("nq", 0) for e in range(5)]
        means = [stats.get(e, {}).get("q") for e in range(5)]
        sens_prior = clamp(0.25 + 0.35 * min(1, f.multistep / 2) + 0.2 * (1 if f.constraints else 0) + 0.15 * min(1, f.verify_req)
                           - 0.3 * min(1, f.ambiguity / 3) - 0.15 * (1 if f.approx_tokens > 3500 else 0), 0.05, 0.95)
        sens = sens_prior
        if n[0] >= 3:
            hi = [means[e] for e in range(1, 5) if n[e] >= 3 and means[e] is not None]
            if hi:
                gain_obs = clamp((max(hi) - means[0]) / max(0.05, 1 - means[0]), 0, 1)
                w = (n[0] + max(n[1:])) / (n[0] + max(n[1:]) + cfg.cold_k)
                sens = w * gain_obs + (1 - w) * sens_prior
        q0p = clamp(0.93 - 0.5 * f.complexity - 0.12 * min(1, f.ambiguity / 3) - 0.08 * f.novelty + bias, 0.05, 0.97)
        q0 = (n[0] * means[0] + cfg.cold_k * q0p) / (n[0] + cfg.cold_k) if n[0] and means[0] is not None else q0p
        head, scale, emul = 1 - q0, 0.3 + 0.7 * sens, (0.7 if kind == "emulated" else 1.0)
        q: List[float] = []
        src: List[str] = []
        for e in range(5):
            g = self.GAIN_FRAC[e] * head * scale * emul
            if e >= 3 and not objective:
                g *= 0.6
            prior = clamp(q0 + g, 0.0, 0.995)
            if e == 0:
                q.append(q0)
                src.append("data" if n[0] >= cfg.cold_k else ("blend" if n[0] else "prior"))
            elif n[e] and means[e] is not None:
                q.append((n[e] * means[e] + cfg.cold_k * prior) / (n[e] + cfg.cold_k))
                src.append("data" if n[e] >= cfg.cold_k else "blend")
            else:
                q.append(prior)
                src.append("prior")
        in_tok = f.approx_tokens
        base_out = clamp(150 + 80 * f.operations + 0.3 * in_tok, 64, 2048)
        if stats.get(0, {}).get("nt"):
            base_out = (stats[0]["nt"] * stats[0]["ot"] + cfg.cold_k * base_out) / (stats[0]["nt"] + cfg.cold_k)
        mult = self.TOK_MULT_NATIVE if kind == "native" else self.TOK_MULT_EMUL
        tps = prof.tps_mean or self.ASSUMED_TPS
        toks, lats, costs, calls = [], [], [], []
        for e in range(5):
            st = stats.get(e, {})
            t = base_out * mult[e]
            cl = float(self.CALLS[e])
            lt = t / tps + 1.0 * cl
            if st.get("nt") and st.get("ot") is not None:
                w = st["nt"] / (st["nt"] + cfg.cold_k)
                t = w * st["ot"] + (1 - w) * t
                cl = w * (st.get("calls") or cl) + (1 - w) * cl
                if st.get("lat"):
                    lt = w * (st["lat"] / 1000.0) + (1 - w) * lt
            toks.append(t)
            lats.append(lt)
            calls.append(max(1, int(round(cl))))
            costs.append(((t + 0.25 * in_tok * cl) / 1000.0 * cfg.w_tok + lt * cfg.w_lat) * res_cost)
        total = sum(n)
        covered = sum(1 for x in n if x >= 3)
        conf = (total / (total + 2 * cfg.cold_k)) * (0.4 + 0.6 * covered / 5.0)
        phase = "COLD"
        if total >= cfg.min_samples:
            phase = "CALIBRATED"
        if cfg.learning_enabled and total >= cfg.learned_min and sum(1 for x in n if x >= 5) >= 3:
            phase = "LEARNED"
        return Curve(q, src, n, toks, lats, costs, calls, sens, phase, min(0.99, conf), kind, not prof.tps_mean)

    def _utility(self, cv: Curve, e: int, budget: Budget) -> float:
        p_exceed = clamp((cv.latency[e] / budget.max_elapsed - 0.5) * 2, 0.0, 1.0)
        return cv.q[e] - self.cfg.lam * cv.cost[e] - self.cfg.w_risk * (1 - cv.q[e]) * p_exceed

    def feasible(self, cv: Curve, budget: Budget) -> int:
        """Highest effort whose projected plan fits the budget envelope (contiguous from E0)."""
        top = 0
        for e in range(5):
            ver_cost = cv.cost[e] - cv.cost[min(e, 2)] if e >= 3 else 0.0
            if cv.calls[e] > budget.allowed_calls or cv.tokens[e] > budget.max_tokens or cv.latency[e] > budget.max_elapsed * 0.8 \
                    or (e >= 3 and ver_cost > budget.max_verification_cost):
                break
            top = e
        return top

    def decide(self, f: RequestFeatures, prof: ModelProfile, budget: Budget, res: ResourceSnapshot, objective: bool,
               prev_effort: Optional[int] = None, trace_id: str = "", forced: Optional[int] = None) -> EffortDecision:
        cfg = self.cfg
        lim = ResourceGuard.limits(res.state if res.observed else "NORMAL")
        cv = self.curve(f, prof, objective, lim["cost"])
        reasons: List[str] = []
        cap = min(budget.max_effort, lim["cap"], self.feasible(cv, budget))
        if cap < budget.max_effort:
            reasons.append(f"cap E{cap} (budget/resource: {res.state if res.observed else 'budget'})")
        U = [self._utility(cv, e, budget) for e in range(5)]
        MV = [0.0] + [(cv.q[e] - cv.q[e - 1]) / max(1e-6, cv.cost[e] - cv.cost[e - 1]) for e in range(1, 5)]
        explored = False
        if forced is not None:
            e = int(clamp(forced, 0, budget.max_effort))
            reasons.append(f"fixed effort E{forced}" + (f" clamped to E{e} by budget" if e != forced else ""))
        else:
            best = max(range(cap + 1), key=lambda i: U[i])
            e = min(i for i in range(cap + 1) if U[i] >= U[best] - cfg.util_margin)
            reasons.append(f"minimum sufficient effort: E{e} (utility {U[e]:.3f}, best {U[best]:.3f})")
            if cv.phase == "LEARNED":
                pick = self._ucb(prof, f, cv, cap)
                if pick is not None and pick != e:
                    reasons.append(f"learned policy (UCB) prefers E{pick}")
                    e = pick
            if e > 0 and MV[e] < cfg.mv_min:
                reasons.append(f"marginal value {MV[e]:.3f} below threshold")
            if prev_effort is not None and prev_effort != e:
                unc = 1.0 - cv.q[min(prev_effort, 4)]
                if e > prev_effort and unc < cfg.enter_thr:
                    reasons.append(f"hysteresis: uncertainty {unc:.2f} < enter {cfg.enter_thr}; staying E{prev_effort}")
                    e = prev_effort
                elif e < prev_effort and unc > cfg.exit_thr:
                    reasons.append(f"hysteresis: uncertainty {unc:.2f} > exit {cfg.exit_thr}; staying E{prev_effort}")
                    e = min(prev_effort, cap)
            e = min(e, cap)
            rnd = int(hashlib.sha256((trace_id or "x").encode()).hexdigest()[:8], 16) / 4294967296.0
            eps = cfg.explore_eps * (1 - cv.confidence) + (0.0 if cv.phase == "COLD" else 0.0)
            if rnd < eps and (not res.observed or res.state == "NORMAL"):
                nb = [x for x in (e - 1, e + 1) if 0 <= x <= cap and U[e] - U[x] <= 0.08]
                if nb:
                    e2 = min(nb, key=lambda x: (cv.n[x], x))
                    reasons.append(f"bounded exploration of E{e2}")
                    e, explored = e2, True
        mv_sel = MV[e] if e > 0 else 0.0
        return EffortDecision(trace_id=trace_id, model=prof.model_id, family=f.family, effort=e, mode=prof.mode_label(e),
                              native_value=prof.native_value(e), phase=cv.phase, confidence=cv.confidence, sensitivity=cv.sens,
                              expected_quality=cv.q[e], expected_incremental_cost=cv.cost[e] - cv.cost[0], marginal_value=mv_sel,
                              q=cv.q, cost=cv.cost, utility=U, marginal=MV, cap=cap,
                              resource_state=res.state if res.observed else NA, reasons=reasons, explored=explored, kind=cv.kind, sig=f.sig)

    def _ucb(self, prof: ModelProfile, f: RequestFeatures, cv: Curve, cap: int) -> Optional[int]:
        """Contextual-bandit style pick: bins = task family x model x kind (the context); reward = quality - lam*cost."""
        stats = self.profiles.effort_stats(prof.model_id, f.family, cv.kind)
        total = sum(cv.n)
        best, best_s = None, -1e9
        for e in range(cap + 1):
            if cv.n[e] < 3 or stats.get(e, {}).get("q") is None:
                continue
            s = stats[e]["q"] - self.cfg.lam * cv.cost[e] + self.cfg.ucb_c * math.sqrt(math.log(max(2, total)) / cv.n[e])
            if s > best_s:
                best, best_s = e, s
        return best


class Hysteresis:
    def __init__(self, store: Store, ttl: float = 3600.0):
        self.store, self.ttl = store, ttl

    def get(self, key: str) -> Optional[int]:
        with self.store.read() as c:
            r = c.execute("SELECT effort, updated_at FROM effort_state WHERE key=?", (key,)).fetchone()
        return r["effort"] if r and now() - r["updated_at"] < self.ttl else None

    def set(self, key: str, effort: int) -> None:
        with self.store.tx() as c:
            c.execute("INSERT OR REPLACE INTO effort_state(key,effort,updated_at) VALUES(?,?,?)", (key, effort, now()))


# =====================================================================================
# LM STUDIO BRIDGE (official local REST only; never loads/unloads/downloads/switches models)
# =====================================================================================
@dataclasses.dataclass
class ChatRequest:
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 1024
    temperature: float = 0.2
    reasoning_effort: Optional[str] = None
    timeout: float = 120.0


@dataclasses.dataclass
class ModelResult:
    text: str
    finish_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None
    tps: Optional[float] = None
    tokens_estimated: bool = False
    effort_param: Optional[str] = None


class ModelBackend:
    def list_models(self) -> Tuple[str, List[Dict[str, Any]]]:
        raise NotImplementedError

    def loaded_state(self, model_id: str) -> str:
        raise NotImplementedError

    def chat(self, req: ChatRequest) -> ModelResult:
        raise NotImplementedError

    def ensure_loaded(self, model_id: str) -> None:
        st = self.loaded_state(model_id)
        if st != "loaded":
            raise PlusError("model_not_loaded" if st == "not-loaded" else "model_state_unknown",
                            f"model '{model_id[:60]}' is not reported as loaded by LM Studio; Nythos Plus never loads models "
                            f"(load it yourself in LM Studio)")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class LMStudioBridge(ModelBackend):
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base = SecurityGuard.local_url(base_url)
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _get(self, path: str, timeout: float = 3.0) -> Any:
        req = urllib.request.Request(self.base + path, headers={"Accept": "application/json"})
        with self.opener.open(req, timeout=timeout) as r:
            return json.loads(r.read(5_000_000).decode("utf-8"))

    def list_models(self) -> Tuple[str, List[Dict[str, Any]]]:
        try:
            d = self._get("/api/v0/models")
            rows = d.get("data", []) if isinstance(d, dict) else []
            return "native_v0", [r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)]
        except urllib.error.HTTPError:
            pass
        except (urllib.error.URLError, OSError, ValueError):
            return "unavailable", []
        try:
            d = self._get("/v1/models")
            return "openai_v1", [{"id": r["id"], "state": "unknown"} for r in d.get("data", []) if isinstance(r, dict) and isinstance(r.get("id"), str)]
        except (urllib.error.URLError, OSError, ValueError):
            return "unavailable", []

    def loaded_state(self, model_id: str) -> str:
        flavor, rows = self.list_models()
        if flavor != "native_v0":
            return "unknown"
        for r in rows:
            if r["id"] == model_id:
                return "loaded" if r.get("state") == "loaded" else "not-loaded"
        return "not-loaded"

    def reachable(self) -> bool:
        return self.list_models()[0] != "unavailable"

    @staticmethod
    def capabilities_of(row: Dict[str, Any]) -> Dict[str, Any]:
        caps = row.get("capabilities") if isinstance(row.get("capabilities"), list) else []
        return {"type": row.get("type"), "arch": row.get("arch"), "state": row.get("state"),
                "max_context": row.get("max_context_length"), "tool_use": "tool_use" in caps,
                "vision": row.get("type") == "vlm"}

    def chat(self, req: ChatRequest) -> ModelResult:
        body: Dict[str, Any] = {"model": req.model, "messages": req.messages, "max_tokens": req.max_tokens,
                                "temperature": req.temperature, "stream": True, "stream_options": {"include_usage": True}}
        if req.reasoning_effort:
            body["reasoning_effort"] = req.reasoning_effort
        r = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                   headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.monotonic()
        deadline = t0 + req.timeout
        try:
            resp = self.opener.open(r, timeout=min(req.timeout, 60))
        except urllib.error.HTTPError as e:
            raise PlusError(f"http_{e.code}", f"LM Studio rejected the request (HTTP {e.code})")
        except (urllib.error.URLError, OSError) as e:
            raise PlusError("lm_unavailable", f"LM Studio not reachable: {type(e).__name__}")
        text, rchars, first, usage, finish = [], 0, None, None, None
        try:
            ctype = resp.headers.get("Content-Type", "")
            if "event-stream" in ctype:
                total = 0
                for raw in resp:
                    if time.monotonic() > deadline:
                        finish = "timeout"
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        ch = json.loads(payload)
                    except ValueError:
                        continue
                    if isinstance(ch.get("usage"), dict):
                        usage = ch["usage"]
                    for c in ch.get("choices") or []:
                        d = c.get("delta") or {}
                        piece, rs = d.get("content") or "", (d.get("reasoning_content") or d.get("reasoning") or "")
                        if (piece or rs) and first is None:
                            first = time.monotonic()
                        if piece:
                            text.append(piece)
                            total += len(piece)
                        rchars += len(rs)
                        if c.get("finish_reason"):
                            finish = c["finish_reason"]
                    if total > Hard.MAX_OUTPUT_EVAL:
                        finish = "client_limit"
                        break
            else:
                d = json.loads(resp.read(10_000_000).decode("utf-8"))
                first = time.monotonic()
                c0 = (d.get("choices") or [{}])[0]
                text.append((c0.get("message") or {}).get("content") or "")
                finish, usage = c0.get("finish_reason"), d.get("usage")
        except (OSError, ValueError) as e:
            raise PlusError("lm_stream_error", f"stream failed: {type(e).__name__}")
        finally:
            resp.close()
        t1 = time.monotonic()
        out = "".join(text)
        est = usage is None or "completion_tokens" not in (usage or {})
        otok = int((len(out) + rchars) / 4) if est else int(usage["completion_tokens"])
        rt = None
        if usage and isinstance(usage.get("completion_tokens_details"), dict) and usage["completion_tokens_details"].get("reasoning_tokens") is not None:
            rt = int(usage["completion_tokens_details"]["reasoning_tokens"])
        tps = None if est or first is None or (t1 - first) < 0.05 else otok / (t1 - first)
        return ModelResult(out, finish, None if not usage else usage.get("prompt_tokens"), otok, rt, (t1 - t0) * 1000.0,
                           None if first is None else (first - t0) * 1000.0, tps, est, req.reasoning_effort)


def probe_native_effort(backend: ModelBackend, model: str, values: Optional[List[str]] = None) -> Dict[str, Any]:
    """Explicit user-requested probe (two tiny inference calls): does a native effort parameter change behaviour?"""
    values = values or ["low", "medium", "high"]
    backend.ensure_loaded(model)
    prompt = "A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much is the ball? Reply with the number only."
    res = {}
    try:
        for v in (values[0], values[-1]):
            res[v] = backend.chat(ChatRequest(model, [{"role": "user", "content": prompt}], max_tokens=1500, reasoning_effort=v, timeout=120))
    except PlusError as e:
        if e.code.startswith("http_4"):
            return {"state": "UNSUPPORTED", "values": [], "detail": e.message}
        raise
    lo, hi = res[values[0]], res[values[-1]]
    a = lo.reasoning_tokens if lo.reasoning_tokens is not None and hi.reasoning_tokens is not None else lo.output_tokens
    b = hi.reasoning_tokens if lo.reasoning_tokens is not None and hi.reasoning_tokens is not None else hi.output_tokens
    basis = "reasoning_tokens" if lo.reasoning_tokens is not None and hi.reasoning_tokens is not None else "output_tokens(estimate)"
    if a is not None and b is not None and b >= max(a * 1.15, a + 8):
        return {"state": "VERIFIED", "values": values, "detail": f"{values[0]}={a} -> {values[-1]}={b} via {basis}"}
    return {"state": "INEFFECTIVE", "values": [], "detail": f"no measurable change ({values[0]}={a}, {values[-1]}={b} via {basis}); treated as emulated"}


# =====================================================================================
# EXECUTION: budget tracker, effort executor, pulse runner
# =====================================================================================
class BudgetTracker:
    def __init__(self, b: Budget):
        self.b, self.t0 = b, time.monotonic()
        self.calls = self.out_tokens = self.escalations = 0
        self.reason_tokens, self.reason_known, self.verify_cost = 0, False, 0.0

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def remaining_time(self) -> float:
        return self.b.max_elapsed - self.elapsed()

    def remaining_tokens(self) -> int:
        return self.b.max_tokens - self.out_tokens

    def can_afford(self, calls: int, tokens: float) -> bool:
        return (self.calls + calls <= self.b.allowed_calls and self.remaining_tokens() >= min(tokens, 64) * 0.5
                and self.remaining_time() > 2.0 and not self.reasoning_exceeded())

    def reasoning_exceeded(self) -> bool:
        return self.b.max_reasoning_tokens is not None and self.reason_known and self.reason_tokens > self.b.max_reasoning_tokens

    def limits(self) -> Tuple[int, float]:
        if self.calls >= self.b.allowed_calls or self.remaining_tokens() < 32 or self.remaining_time() < 1.0 or self.reasoning_exceeded():
            raise BudgetStop("budget envelope exhausted")
        return min(self.b.max_tokens_per_call, self.remaining_tokens()), self.remaining_time()

    def charge(self, r: ModelResult) -> None:
        self.calls += 1
        self.out_tokens += r.output_tokens or 0
        if r.reasoning_tokens is not None:
            self.reason_known = True
            self.reason_tokens += r.reasoning_tokens


@dataclasses.dataclass
class Attempt:
    effort: int
    text: str
    results: List[ModelResult]
    verifier: Optional[str] = None
    repairs: int = 0
    clean: bool = True
    label: str = ""


class EffortExecutor:
    HINT_E1 = "Check the stated requirements one by one before answering. Reply with the final answer only."
    HINT_PLAN = ("Work in two phases. Phase 1: under the heading PLAN: write a short plan (at most 8 lines). "
                 "Phase 2: under the heading ANSWER: write the final answer only.")
    VERIFIER = ("You are a strict verifier. Judge ONLY against the requirements stated in the task. Write one line per violated "
                "requirement (or 'none'), then on the last line exactly: VERDICT: PASS or VERDICT: FAIL.")

    def __init__(self, backend: ModelBackend, cfg: Config):
        self.backend, self.cfg = backend, cfg

    def _call(self, model: str, messages: List[Dict[str, str]], tracker: BudgetTracker, native: Optional[str], max_tokens: Optional[int] = None) -> ModelResult:
        mt, to = tracker.limits()
        r = self.backend.chat(ChatRequest(model, messages, min(mt, max_tokens or mt), 0.2, native, min(self.cfg.http_timeout, to)))
        tracker.charge(r)
        return r

    @staticmethod
    def _msgs(system: Optional[str], hint: Optional[str], user: str) -> List[Dict[str, str]]:
        s = "\n".join(x for x in (system, hint) if x)
        return ([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": user}]

    def generate(self, model: str, prompt: str, system: Optional[str], effort: Optional[int], prof: ModelProfile,
                 tracker: BudgetTracker, feedback: Optional[List[str]] = None) -> Attempt:
        """effort None = RAW (model's normal behaviour: no hints, no effort parameter)."""
        if effort is None:
            r = self._call(model, self._msgs(system, None, prompt), tracker, None)
            return Attempt(-1, r.text, [r], label="RAW")
        native = prof.native_value(effort)
        hint = None
        if effort == 1:
            hint = self.HINT_E1
        elif effort >= 2 and prof.kind() == "emulated":
            hint = self.HINT_PLAN
        user = prompt
        if feedback:
            user += "\n\nYour previous answer violated these requirements: " + "; ".join(feedback[:6]) + ". Produce a corrected final answer."
        r = self._call(model, self._msgs(system, hint, user), tracker, native)
        text = r.text
        if hint == self.HINT_PLAN:
            m = re.search(r"ANSWER:\s*(.*)", text, re.S | re.I)
            text = m.group(1).strip() if m else text
        att = Attempt(effort, text, [r], clean=feedback is None, label=EFFORT_NAMES[effort])
        if effort >= 3:
            att.verifier = self.verify(model, prompt, text, system, prof, tracker, att)
        return att

    def verify(self, model: str, prompt: str, answer: str, system: Optional[str], prof: ModelProfile, tracker: BudgetTracker, att: Attempt) -> Optional[str]:
        try:
            r = self._call(model, [{"role": "system", "content": self.VERIFIER},
                                   {"role": "user", "content": f"TASK:\n{prompt[:6000]}\n\nCANDIDATE ANSWER:\n{answer[:6000]}"}],
                           tracker, prof.native_value(1), 300)
        except BudgetStop:
            return None
        att.results.append(r)
        v = re.findall(r"VERDICT:\s*(PASS|FAIL)", r.text, re.I)
        return v[-1].upper() if v else "UNPARSED"

    def repair(self, model: str, prompt: str, system: Optional[str], prev: str, violations: List[str], prof: ModelProfile,
               tracker: BudgetTracker) -> ModelResult:
        user = (f"{prompt}\n\nPrevious answer:\n{prev[:4000]}\n\nIt violated: " + "; ".join(violations[:6] or ["unspecified requirements"]) +
                ". Produce a corrected final answer only.")
        return self._call(model, self._msgs(system, None, user), tracker, prof.native_value(2))


class HistoryStore:
    def __init__(self, store: Store, cfg: Config):
        self.store, self.cfg = store, cfg

    def save_decision(self, d: EffortDecision, status: str = "pending") -> None:
        with self.store.tx() as c:
            c.execute("INSERT OR REPLACE INTO decisions(trace_id,created_at,model,family,effort,mode,phase,policy_confidence,kind,sig,pred_q,payload,status) "
                      "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (d.trace_id, now(), d.model, d.family, d.effort, d.mode, d.phase, d.confidence,
                                                              d.kind, d.sig, d.expected_quality, json.dumps(d.public()), status))

    def record_run(self, row: Dict[str, Any], attempts: List[Dict[str, Any]]) -> None:
        with self.store.tx() as c:
            cols = list(row)
            c.execute(f"INSERT INTO outcomes({','.join(cols)}) VALUES({','.join('?' * len(cols))})", [row[k] for k in cols])
            for a in attempts:
                ac = list(a)
                c.execute(f"INSERT INTO attempts({','.join(ac)}) VALUES({','.join('?' * len(ac))})", [a[k] for k in ac])
            c.execute("UPDATE decisions SET status='evaluated' WHERE trace_id=?", (row["trace_id"],))

    def recent(self, limit: int = 20, bench_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.store.read() as c:
            q, a = "SELECT * FROM outcomes", []
            if bench_id:
                q += " WHERE bench_id=?"
                a.append(bench_id)
            q += " ORDER BY created_at DESC LIMIT ?"
            a.append(int(clamp(limit, 1, 500)))
            return [dict(r) for r in c.execute(q, a)]

    def metrics(self, model: Optional[str] = None, run_mode: Optional[str] = None, bench_id: Optional[str] = None) -> Dict[str, Any]:
        where, args = ["1=1"], []
        for col, v in (("model", model), ("run_mode", run_mode), ("bench_id", bench_id)):
            if v:
                where.append(f"{col}=?")
                args.append(v)
        w = " AND ".join(where)
        sc = f"quality IS NOT NULL AND evidence>={self.cfg.min_evidence}"
        with self.store.read() as c:
            r = c.execute(f"SELECT COUNT(*) n, SUM(CASE WHEN {sc} THEN 1 ELSE 0 END) ns, SUM(CASE WHEN {sc} THEN quality ELSE 0 END) sq, "
                          f"SUM(CASE WHEN {sc} AND quality>=0.999 THEN 1 ELSE 0 END) succ, SUM(cost) cost, "
                          f"SUM(CASE WHEN quality>=0.999 AND {sc} THEN cost ELSE 0 END) cost_succ, SUM(output_tokens) tok, SUM(latency_ms) lat, "
                          f"SUM(repair) rep, SUM(CASE WHEN escalations>0 THEN 1 ELSE 0 END) esc, AVG(final_effort) avg_e, "
                          f"SUM(CASE WHEN {sc} THEN output_tokens ELSE 0 END) tok_s, SUM(CASE WHEN {sc} THEN latency_ms ELSE 0 END) lat_s "
                          f"FROM outcomes WHERE {w}", args).fetchone()
            fp = c.execute(f"SELECT SUM(CASE WHEN a.quality>=0.999 AND a.evidence>={self.cfg.min_evidence} THEN 1 ELSE 0 END) ok, COUNT(*) n "
                           f"FROM attempts a JOIN outcomes o ON o.trace_id=a.trace_id WHERE a.idx=0 AND {w.replace('model', 'o.model').replace('run_mode', 'o.run_mode').replace('bench_id', 'o.bench_id')}", args).fetchone()
            vy = c.execute(f"SELECT COUNT(*) n, SUM(CASE WHEN quality>(SELECT a2.quality FROM attempts a2 WHERE a2.trace_id=outcomes.trace_id ORDER BY a2.idx LIMIT 1) THEN 1 ELSE 0 END) y "
                           f"FROM outcomes WHERE verification IS NOT NULL AND verification!='NONE' AND {w}", args).fetchone()
        n, ns = r["n"] or 0, r["ns"] or 0
        div = lambda a, b: None if not b else a / b
        return {"runs": n, "scored_runs": ns, "mean_quality": div(r["sq"] or 0, ns),
                "effort_efficiency": div(r["sq"] or 0, r["cost"] if ns else 0),
                "first_pass_success": div(fp["ok"] or 0, fp["n"]), "repair_rate": div(r["rep"] or 0, n),
                "escalation_rate": div(r["esc"] or 0, n), "verification_yield": div(vy["y"] or 0, vy["n"]),
                "avg_cost_per_success": div(r["cost_succ"] or 0, r["succ"]), "success": r["succ"] or 0,
                "quality_per_1k_tokens": div((r["sq"] or 0) * 1000.0, r["tok_s"]), "quality_per_second": div((r["sq"] or 0) * 1000.0, r["lat_s"]),
                "avg_final_effort": r["avg_e"], "total_cost_units": r["cost"]}


class PulseRunner:
    def __init__(self, engine: "Engine", backend: ModelBackend):
        self.e, self.backend = engine, backend
        self.exec = EffortExecutor(backend, engine.cfg)

    def run(self, prompt: str, model: str, *, checks: Optional[List[Dict[str, Any]]] = None, mode: str = "PULSE",
            budget: Optional[Dict[str, Any]] = None, session: str = "default", system: Optional[str] = None,
            bench_id: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
        cfg = self.e.cfg
        if mode not in RUN_MODES or mode == "ORACLE":
            raise PlusError("invalid_argument", "mode must be RAW, PULSE or FIXED_E0..FIXED_E4 (ORACLE is a benchmark comparison)")
        prompt = SecurityGuard.text(prompt, "prompt", Hard.MAX_PROMPT)
        SecurityGuard.ident(model, "model")
        with RecursionGuard():
            res = self.e.resource.sample(force=True)
            lim = ResourceGuard.limits(res.state if res.observed else "NORMAL")
            if not lim["allow"]:
                raise PlusError("resource_critical", "system resources are CRITICAL; no new expensive work is started")
            self.backend.ensure_loaded(model)
            all_checks = [dict(c, source=c.get("source", "user")) for c in (checks or [])] + OutcomeEvaluator.auto_checks(prompt)
            b = cfg.make_budget(budget)
            b.max_effort = min(b.max_effort, lim["cap"] if mode == "PULSE" else b.max_effort)
            b.max_escalations = min(b.max_escalations, lim["esc"])
            feats = self.e.features(prompt, model)
            prof = self.e.profiles.get(model)
            trace = new_id("t")
            hkey = f"{model}|{feats.family}|{session}"
            forced = int(mode[-1]) if mode.startswith("FIXED_E") else None
            dec = self.e.policy.decide(feats, prof, b, res, bool(all_checks), self.e.hyst.get(hkey) if mode == "PULSE" else None, trace, forced)
            raw = mode == "RAW"
            if raw:
                dec.effort, dec.mode = 0, "RAW"
            self.e.history.save_decision(dec)
            tr = BudgetTracker(b)
            slot = f"model:{model}"
            self.e.store.acquire(slot, trace, b.max_elapsed + 30)
            try:
                return self._loop(prompt, model, system, all_checks, mode, raw, b, tr, feats, prof, dec, res, hkey, bench_id, task_id)
            finally:
                self.e.store.release(slot, trace)

    def _loop(self, prompt, model, system, checks, mode, raw, b, tr, feats, prof, dec, res, hkey, bench_id, task_id) -> Dict[str, Any]:
        cfg, ev = self.e.cfg, self.e.evaluator
        e: Optional[int] = None if raw else dec.effort
        attempts: List[Tuple[Attempt, Outcome]] = []
        steps: List[List[Any]] = []
        feedback, stop_reason = None, "single_pass"
        while True:
            try:
                att = self.exec.generate(model, prompt, system, e, prof, tr, feedback)
            except BudgetStop:
                stop_reason = "budget_exhausted"
                break
            out = ev.evaluate(att.text, att.results[0].finish_reason, checks, dec.expected_quality, att.verifier)
            if (e or 0) >= 4 and not raw:
                while out.violations and att.repairs < min(2, max(1, b.max_extra_passes)):
                    try:
                        r = self.exec.repair(model, prompt, system, att.text, out.violations, prof, tr)
                    except BudgetStop:
                        break
                    att.results.append(r)
                    att.repairs += 1
                    att.text = r.text
                    out = ev.evaluate(att.text, r.finish_reason, checks, dec.expected_quality, att.verifier)
            attempts.append((att, out))
            steps.append([e if e is not None else "RAW", round(out.pass_fraction, 2), round(out.evidence_strength, 2), "evaluated"])
            if mode != "PULSE":
                break
            nxt, why = self._next(e, out, dec, tr, b, prof, feats)
            stop_reason = why
            steps[-1][3] = why if nxt is None else f"escalate->E{nxt}"
            if nxt is None:
                break
            tr.escalations += 1
            feedback = out.violations or ["requirements not satisfied"]
            e = nxt
        if not attempts:
            raise PlusError("budget_exhausted", "budget envelope forbids even the first call")
        dis = DissentTracker()
        for a, o in attempts:
            dis.add(f"E{a.effort}", a.text, o.model_confidence, o.evidence_strength, a.verifier,
                    (a.verifier == "FAIL" and o.pass_fraction >= 0.999) or (a.verifier == "PASS" and o.pass_fraction < 0.5 and o.evidence_strength >= 0.5))
        dscore = dis.score()
        best_i = max(range(len(attempts)), key=lambda i: ((attempts[i][1].quality if attempts[i][1].quality is not None else -1.0),
                                                           attempts[i][1].system_confidence, -i))
        if mode == "PULSE" and dscore >= cfg.dissent_thr and tr.escalations < b.max_escalations and tr.can_afford(1, 300):
            a = attempts[best_i][0]
            v = self.exec.verify(model, prompt, a.text, system, prof, tr, a)
            if v:
                a.verifier = v
                stop_reason += "+dissent_check"
        att, out = attempts[best_i]
        tot_calls = sum(len(a.results) for a, _ in attempts)
        results = [r for a, _ in attempts for r in a.results]
        in_t = sum(r.input_tokens or 0 for r in results) if any(r.input_tokens is not None for r in results) else None
        out_t = sum(r.output_tokens or 0 for r in results) if any(r.output_tokens is not None for r in results) else None
        rt = sum(r.reasoning_tokens or 0 for r in results) if any(r.reasoning_tokens is not None for r in results) else None
        lat = sum(r.latency_ms for r in results)
        ttft = next((r.ttft_ms for r in results if r.ttft_ms is not None), None)
        tps_l = [r.tps for r in results if r.tps]
        tps = statistics.mean(tps_l) if tps_l else None
        cost = ((out_t or 0) + 0.25 * (in_t or feats.approx_tokens * tot_calls)) / 1000.0 * cfg.w_tok + lat / 1000.0 * cfg.w_lat
        scored = out.quality is not None and out.evidence_strength >= cfg.min_evidence
        ver = next((a.verifier for a, _ in attempts if a.verifier), None)
        row = {"trace_id": dec.trace_id, "created_at": now(), "bench_id": bench_id, "task_id": task_id, "run_mode": mode, "model": model,
               "family": feats.family, "effort": None if raw else dec.effort, "final_effort": None if raw else attempts[best_i][0].effort,
               "kind": "raw" if raw else dec.kind, "input_tokens": in_t, "output_tokens": out_t, "reasoning_tokens": rt, "latency_ms": lat,
               "ttft_ms": ttft, "tps": tps, "verification": ver or "NONE", "quality": out.quality if scored else None,
               "evidence": out.evidence_strength, "success": (1 if out.quality is not None and out.quality >= 0.999 else 0) if scored else None,
               "repair": 1 if any(a.repairs for a, _ in attempts) or any(not a.clean for a, _ in attempts) else 0,
               "escalations": tr.escalations, "calls": tot_calls, "cost": cost, "resource_state": res.state if res.observed else NA,
               "policy_confidence": dec.confidence, "steps": json.dumps(steps)}
        arows = []
        for i, (a, o) in enumerate(attempts):
            rs = a.results
            arows.append({"attempt_id": new_id("a"), "trace_id": dec.trace_id, "idx": i, "model": model, "family": feats.family,
                          "kind": "raw" if raw else dec.kind, "effort": max(0, a.effort), "clean": 1 if (a.clean and not raw) else 0,
                          "quality": o.quality if (o.quality is not None and o.evidence_strength >= cfg.min_evidence) else None,
                          "evidence": o.evidence_strength, "pred_q": dec.q[max(0, min(4, a.effort))], "sig": feats.sig,
                          "input_tokens": sum(r.input_tokens or 0 for r in rs) if any(r.input_tokens is not None for r in rs) else None,
                          "output_tokens": sum(r.output_tokens or 0 for r in rs) if any(r.output_tokens is not None for r in rs) else None,
                          "reasoning_tokens": sum(r.reasoning_tokens or 0 for r in rs) if any(r.reasoning_tokens is not None for r in rs) else None,
                          "latency_ms": sum(r.latency_ms for r in rs), "ttft_ms": next((r.ttft_ms for r in rs if r.ttft_ms is not None), None),
                          "calls": len(rs), "verifier": a.verifier, "created_at": now()})
        self.e.history.record_run(row, arows)
        if mode == "PULSE":
            self.e.hyst.set(hkey, attempts[best_i][0].effort)
        return {"text": att.text, "trace_id": dec.trace_id, "decision": dec.public(), "outcome": out.public(), "stop_reason": stop_reason,
                "dissent": round(dscore, 3), "steps": steps, "escalations": tr.escalations, "verification": ver or "NONE",
                "usage": {"calls": tot_calls, "input_tokens": NA if in_t is None else in_t, "output_tokens": NA if out_t is None else out_t,
                          "reasoning_tokens": NA if rt is None else rt, "ttft_ms": NA if ttft is None else round(ttft, 1),
                          "tokens_per_sec": NA if tps is None else round(tps, 2), "elapsed_s": round(tr.elapsed(), 2), "cost_units": round(cost, 4)},
                "learned_from": bool(scored), "run_mode": mode, "family": feats.family, "final_effort": None if raw else attempts[best_i][0].effort}

    def _next(self, e: Optional[int], out: Outcome, dec: EffortDecision, tr: BudgetTracker, b: Budget, prof: ModelProfile,
              feats: RequestFeatures) -> Tuple[Optional[int], str]:
        cfg = self.e.cfg
        e = e or 0
        if out.pass_fraction >= 0.999 and out.evidence_strength >= cfg.min_evidence and out.system_confidence >= cfg.stop_conf and not out.issues:
            return None, "sufficient_evidence"
        if e >= min(b.max_effort, dec.cap):
            return None, "max_effort_reached"
        if tr.escalations >= b.max_escalations:
            return None, "max_escalations"
        nxt = e + 1
        scored = out.quality is not None and out.evidence_strength >= cfg.min_evidence
        if not scored:
            ev = verification_value(out.system_confidence, prof.repair_rate, max(dec.cost[3] - dec.cost[2], 0.01), cfg,
                                    b.max_verification_cost - tr.verify_cost)
            if ev["justified"] and e < 3 and tr.can_afford(2, 300) and b.max_effort >= 3 and dec.cap >= 3:
                tr.verify_cost += dec.cost[3] - dec.cost[2]
                return 3, "verification_justified"
            return None, "unverifiable_minimum_sufficient"
        unc = 1.0 - out.system_confidence
        if unc < cfg.enter_thr and not out.violations:
            return None, "below_enter_threshold"
        gain = max(0.0, dec.q[nxt] - (out.quality or 0.0))
        mv = gain / max(1e-6, dec.cost[nxt] - dec.cost[e])
        if mv < cfg.mv_min:
            return None, "diminishing_returns"
        if not tr.can_afford(self.e.policy.CALLS[nxt], dec.cost[nxt] and 200):
            return None, "budget_exhausted"
        return nxt, "escalate"


# =====================================================================================
# ENGINE (composition)
# =====================================================================================
class Engine:
    def __init__(self, paths: Optional[Paths] = None, cfg: Optional[Config] = None, resource: Optional[ResourceGuard] = None):
        self.paths = paths or Paths()
        self.paths.ensure()
        self.cfg = cfg or Config.load(self.paths)
        self.store = Store(self.paths.db)
        self.store.init_schema()
        self.profiles = ProfileStore(self.store, self.cfg)
        self.evaluator = OutcomeEvaluator(self.cfg)
        self.resource = resource or ResourceGuard(thresholds=self.cfg.res_thresholds)
        self.policy = EffortPolicy(self.cfg, self.profiles)
        self.hyst = Hysteresis(self.store)
        self.history = HistoryStore(self.store, self.cfg)

    def features(self, prompt: str, model: str, has_image: bool = False) -> RequestFeatures:
        f = probe_prompt(prompt, has_image)
        with self.store.read() as c:
            f.history_failures = c.execute("SELECT COUNT(*) FROM (SELECT quality FROM attempts WHERE model=? AND family=? AND quality IS NOT NULL "
                                           "ORDER BY created_at DESC LIMIT 20) WHERE quality<0.5", (model, f.family)).fetchone()[0]
            n = c.execute("SELECT COUNT(*) FROM attempts WHERE model=? AND sig=?", (model, f.sig)).fetchone()[0]
        f.novelty = 1.0 / (1.0 + n / 3.0)
        return f

    def advise(self, prompt: str, model: str, session: str = "default", objective: bool = False, budget: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Decision only; makes no model call."""
        f = self.features(prompt, model)
        prof = self.profiles.get(model)
        res = self.resource.sample()
        b = self.cfg.make_budget(budget)
        trace = new_id("t")
        d = self.policy.decide(f, prof, b, res, objective or bool(OutcomeEvaluator.auto_checks(prompt)),
                               self.hyst.get(f"{model}|{f.family}|{session}"), trace)
        self.history.save_decision(d)
        hint = None
        if d.mode == "EMULATED" or (d.mode == "HYBRID"):
            hint = {1: EffortExecutor.HINT_E1, 2: EffortExecutor.HINT_PLAN, 3: EffortExecutor.HINT_PLAN, 4: EffortExecutor.HINT_PLAN}.get(d.effort)
        if d.mode == "NATIVE":
            hint = None
        return {"decision": d.public(), "features": f.public(), "hint": hint, "model_call_made": False}


# =====================================================================================
# BENCHMARK
# =====================================================================================
def builtin_tasks() -> List[Dict[str, Any]]:
    filler = [f"Record {i} notes that the storage room temperature was {10 + i % 7} degrees on shift {i % 4}." for i in range(60)]
    filler.insert(37, "The vault code is 7391.")
    T = lambda tid, dom, prompt, checks: {"id": tid, "domain": dom, "prompt": prompt, "checks": checks}
    return [
        T("r1", "reasoning", "Alice is taller than Bob. Bob is taller than Carol. Dave is shorter than Carol. Who is the second tallest? Answer with the name only.", [{"type": "equals", "value": "Bob"}]),
        T("r2", "reasoning", "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Answer yes or no.", [{"type": "equals", "value": "yes"}]),
        T("m1", "math", "Compute 17*23 + 144/12. Answer with the number only.", [{"type": "number", "value": 403}]),
        T("m2", "math", "A train travels 180 km in 2.5 hours. At the same speed, how many km does it travel in 4 hours? Answer with the number only.", [{"type": "number", "value": 288}]),
        T("m3", "math", "What is 15% of 240 plus 35% of 120? Answer with the number only.", [{"type": "number", "value": 78}]),
        T("c1", "coding", "Write a Python function named is_palindrome(s) that returns True if s reads the same forwards and backwards ignoring case and non-alphanumeric characters. Output only code.", [{"type": "python"}, {"type": "defines", "value": "is_palindrome"}]),
        T("c2", "coding", "Write a Python function named fizzbuzz(n) that returns a list of strings for 1..n using the classic FizzBuzz rules. Output only code.", [{"type": "python"}, {"type": "defines", "value": "fizzbuzz"}, {"type": "contains", "value": "FizzBuzz"}]),
        T("p1", "planning", 'Produce a 3-step plan to prepare a presentation. Return JSON only in the form {"steps": ["...", "...", "..."]}.', [{"type": "json"}, {"type": "json_len", "key": "steps", "value": 3}]),
        T("i1", "instruction", "Write exactly 12 words describing the sea. Use no punctuation.", [{"type": "words", "value": 12}]),
        T("i2", "instruction", "List exactly 4 bullet points about recycling. Each line must start with '- '. Do not include the word 'plastic'.", [{"type": "bullets", "value": 4}, {"type": "not_contains", "value": "plastic"}]),
        T("v1", "verification", "Is this statement correct? '17 is divisible by 3.' Answer only 'correct' or 'incorrect'.", [{"type": "equals", "value": "incorrect"}]),
        T("k1", "consistency", "Answer in JSON only with keys 'a' and 'b': a = the capital of France, b = the capital of Italy.", [{"type": "json_keys", "value": ["a", "b"]}, {"type": "contains", "value": "Paris"}, {"type": "contains", "value": "Rome"}]),
        T("t1", "tool_use", 'Return only a JSON object for a tool call named get_weather with arguments city="Oslo" and unit="celsius", in the form {"name": ..., "arguments": {...}}.', [{"type": "json_keys", "value": ["name", "arguments"]}, {"type": "contains", "value": "Oslo"}]),
        T("l1", "long_context", "Read the log below and answer the question.\n\n" + "\n".join(filler) + "\n\nQuestion: What is the vault code? Answer with the number only.", [{"type": "number", "value": 7391}]),
    ]


class Benchmark:
    def __init__(self, engine: Engine, runner: PulseRunner):
        self.e, self.runner = engine, runner

    def run(self, model: str, modes: List[str], tasks: List[Dict[str, Any]], repeat: int = 1, budget: Optional[Dict[str, Any]] = None,
            progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        bench_id = new_id("b")
        want_oracle = "ORACLE" in modes
        run_modes = [m for m in modes if m != "ORACLE"]
        if want_oracle:
            run_modes += [f"FIXED_E{k}" for k in range(5) if f"FIXED_E{k}" not in run_modes]
        done: List[Dict[str, Any]] = []
        for rep in range(max(1, repeat)):
            for t in tasks:
                per: Dict[str, Dict[str, Any]] = {}
                for m in run_modes:
                    try:
                        r = self.runner.run(t["prompt"], model, checks=t.get("checks"), mode=m, budget=budget, bench_id=bench_id, task_id=f"{t['id']}#{rep}")
                    except PlusError as exc:
                        if progress:
                            progress(f"  {t['id']} {m}: {exc.code}")
                        if exc.code in ("model_not_loaded", "model_state_unknown", "lm_unavailable", "resource_critical"):
                            raise
                        continue
                    per[m] = r
                    done.append(r)
                    if progress:
                        progress(f"  {t['id']}#{rep} {m:9s} q={r['outcome']['quality']} E={r['final_effort']} tokens={r['usage']['output_tokens']}")
                if want_oracle:
                    fixed = [(m, r) for m, r in per.items() if m.startswith("FIXED_E") and r["outcome"]["quality"] != NA]
                    if fixed:
                        m, best = max(fixed, key=lambda mr: (mr[1]["outcome"]["quality"], -mr[1]["usage"]["cost_units"]))
                        self._store_oracle(best, bench_id, f"{t['id']}#{rep}", m)
        return {"bench_id": bench_id, "runs": len(done)}

    def _store_oracle(self, best: Dict[str, Any], bench_id: str, task_id: str, src_mode: str) -> None:
        with self.e.store.tx() as c:
            src = c.execute("SELECT * FROM outcomes WHERE trace_id=?", (best["trace_id"],)).fetchone()
            row = dict(src)
            row.update(trace_id=new_id("t"), run_mode="ORACLE", created_at=now(), bench_id=bench_id, task_id=task_id,
                       steps=json.dumps([["oracle_of", src_mode]]))
            c.execute(f"INSERT INTO outcomes({','.join(row)}) VALUES({','.join('?' * len(row))})", list(row.values()))

    def summary(self, bench_id: Optional[str] = None, model: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for m in RUN_MODES:
            x = self.e.history.metrics(model=model, run_mode=m, bench_id=bench_id)
            if x["runs"]:
                x["mode"] = m
                out.append(x)
        return out


# =====================================================================================
# MCP SERVER (JSON-RPC over STDIO; decision/evaluation only - it never runs inference)
# =====================================================================================
def _S(props: Dict[str, Any], required: Tuple[str, ...] = ()) -> Dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


_MODEL = {"type": "string", "pattern": ID_PATTERN, "maxLength": 128}


class MCPServer:
    def __init__(self, engine: Engine):
        self.e = engine
        self.initialized = False
        self.tools = {
            "nythosplus_status": ("Adaptive-effort status: phase, outcomes stored, resource state. Read-only.", _S({}), self._t_status, True),
            "nythosplus_advise": ("Recommend an effort level (E0-E4) for a prompt. Makes no model call. Returns a trace_id.",
                                  _S({"prompt": {"type": "string", "minLength": 1, "maxLength": Hard.MAX_MCP_PROMPT}, "model_id": _MODEL}, ("prompt", "model_id")), self._t_advise, False),
            "nythosplus_evaluate": ("Evaluate a produced answer against objective checks (contains:x, equals:x, number:N, regex:p, json, json_keys:a,b, python, words:N, ...). "
                                    "Stores an outcome for calibration. The answer text is not stored.",
                                    _S({"trace_id": {"type": "string", "pattern": ID_PATTERN, "maxLength": 64}, "output": {"type": "string", "minLength": 1, "maxLength": 20000},
                                        "checks": {"type": "array", "maxItems": 12, "items": {"type": "string", "maxLength": 300}}}, ("trace_id", "output", "checks")), self._t_evaluate, False),
            "nythosplus_policy": ("Show the learned effort response curve for a model and task family.",
                                  _S({"model_id": _MODEL, "family": {"type": "string", "enum": list(FAMILIES), "maxLength": 20}}, ("model_id",)), self._t_policy, True),
            "nythosplus_history": ("Recent stored outcome metadata (no prompts or outputs).", _S({"limit": {"type": "integer", "minimum": 1, "maximum": 20}}), self._t_history, True),
        }

    def _t_status(self, a):
        with self.e.store.read() as c:
            n = c.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
            m = c.execute("SELECT COUNT(DISTINCT model) FROM outcomes").fetchone()[0]
        return {"version": VERSION, "adaptive_effort": "AUTO", "outcomes": n, "models_seen": m, "resources": self.e.resource.sample().public(),
                "inference_via_mcp": False, "lm_studio_model_control": "never"}

    def _t_advise(self, a):
        return self.e.advise(a["prompt"], a["model_id"])

    def _t_evaluate(self, a):
        with self.e.store.tx() as c:
            d = c.execute("SELECT * FROM decisions WHERE trace_id=?", (a["trace_id"],)).fetchone()
        if not d:
            raise PlusError("not_found", "unknown trace_id")
        if d["status"] != "pending":
            raise PlusError("invalid_state", f"trace already {d['status']}")
        checks = [dict(parse_check(s), source="user") for s in a["checks"]]
        if not checks:
            raise PlusError("invalid_argument", "at least one objective check is required (self-reports are not evidence)")
        out = self.e.evaluator.evaluate(a["output"], None, checks, d["pred_q"] or 0.5)
        scored = out.quality is not None and out.evidence_strength >= self.e.cfg.min_evidence
        row = {"trace_id": a["trace_id"], "created_at": now(), "bench_id": None, "task_id": None, "run_mode": "PULSE", "model": d["model"],
               "family": d["family"], "effort": d["effort"], "final_effort": d["effort"], "kind": d["kind"], "input_tokens": None,
               "output_tokens": None, "reasoning_tokens": None, "latency_ms": None, "ttft_ms": None, "tps": None, "verification": "NONE",
               "quality": out.quality if scored else None, "evidence": out.evidence_strength,
               "success": (1 if out.quality >= 0.999 else 0) if scored else None, "repair": 0, "escalations": 0, "calls": 1, "cost": None,
               "resource_state": NA, "policy_confidence": d["policy_confidence"], "steps": "[]"}
        arow = {"attempt_id": new_id("a"), "trace_id": a["trace_id"], "idx": 0, "model": d["model"], "family": d["family"], "kind": d["kind"],
                "effort": d["effort"], "clean": 1, "quality": row["quality"], "evidence": out.evidence_strength, "pred_q": d["pred_q"], "sig": d["sig"],
                "input_tokens": None, "output_tokens": None, "reasoning_tokens": None, "latency_ms": None, "ttft_ms": None, "calls": 1,
                "verifier": None, "created_at": now()}
        self.e.history.record_run(row, [arow])
        return {"outcome": out.public(), "learned_from": bool(scored), "tokens_latency": NA}

    def _t_policy(self, a):
        fam = a.get("family", "general")
        f = RequestFeatures(family=fam, complexity=0.5, approx_tokens=200, sig=f"{fam}|c2|t2")
        prof = self.e.profiles.get(a["model_id"])
        cv = self.e.policy.curve(f, prof, True)
        return {"model": a["model_id"], "family": fam, "kind": cv.kind, "phase": cv.phase, "samples": cv.n, "quality": [round(x, 3) for x in cv.q],
                "source": cv.q_src, "cost": [round(x, 3) for x in cv.cost], "note": "neutral reference task (complexity 0.5); real decisions use the actual prompt"}

    def _t_history(self, a):
        rows = self.e.history.recent(a.get("limit", 10))
        keep = ("trace_id", "run_mode", "model", "family", "effort", "final_effort", "quality", "success", "escalations", "resource_state")
        return {"runs": [{k: r[k] for k in keep} for r in rows]}

    @staticmethod
    def _err(rid, code, msg):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}

    def handle_bytes(self, raw: bytes) -> Optional[Dict[str, Any]]:
        try:
            msg = json.loads(raw.decode("utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError("nonfinite")))
        except (ValueError, UnicodeDecodeError, RecursionError):
            return self._err(None, -32700, "Parse error")
        return self.handle_message(msg)

    def handle_message(self, msg: Any) -> Optional[Dict[str, Any]]:
        if isinstance(msg, list):
            return self._err(None, -32600, "Invalid Request: batches are not supported")
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self._err(None, -32600, "Invalid Request")
        has_id, rid = "id" in msg, msg.get("id")
        if has_id and (isinstance(rid, bool) or not isinstance(rid, (str, int))):
            return self._err(None, -32600, "Invalid Request: bad id")
        method, params = msg.get("method"), msg.get("params") or {}
        if method is None or not has_id:
            return None
        if not isinstance(method, str) or not isinstance(params, dict):
            return self._err(rid, -32600, "Invalid Request")
        ok = lambda r: {"jsonrpc": "2.0", "id": rid, "result": r}
        try:
            if method == "initialize":
                v = params.get("protocolVersion")
                self.initialized = True
                return ok({"protocolVersion": v if v in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0], "capabilities": {"tools": {"listChanged": False}},
                           "serverInfo": {"name": "nythosplus", "title": "Nythos Plus adaptive effort", "version": VERSION},
                           "instructions": "Adaptive-effort advisor. It never loads/unloads models and never runs inference through MCP. Results are advice based on observable outcomes."})
            if method == "ping":
                return ok({})
            if not self.initialized:
                return self._err(rid, -32002, "Server not initialized")
            if method == "tools/list":
                return ok({"tools": [{"name": n, "description": d, "inputSchema": s, "annotations": {"readOnlyHint": ro, "destructiveHint": False, "openWorldHint": False}}
                                     for n, (d, s, _, ro) in self.tools.items()]})
            if method == "tools/call":
                name = params.get("name")
                if not isinstance(name, str) or name not in self.tools:
                    return self._err(rid, -32602, "Unknown tool")
                return ok(self._call(name, params.get("arguments")))
            return self._err(rid, -32601, "Method not found")
        except Exception:
            LOG.error("internal error: %s", traceback.format_exc(limit=3))
            return self._err(rid, -32603, "Internal error")

    def _call(self, name: str, arguments: Any) -> Dict[str, Any]:
        rid = uuid.uuid4().hex
        desc, schema, handler, _ = self.tools[name]
        code = ""
        try:
            body = {"ok": True, "request_id": rid, "result": handler(SecurityGuard.validate(schema, arguments))}
        except PlusError as e:
            code = e.code
            body = {"ok": False, "request_id": rid, "error": {"code": e.code, "message": e.message}}
        except sqlite3.Error:
            code = "db_error"
            body = {"ok": False, "request_id": rid, "error": {"code": "db_error", "message": "database operation failed"}}
        text = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        if len(text) > Hard.MAX_RESULT_CHARS:
            code, text = "output_too_large", json.dumps({"ok": False, "request_id": rid, "error": {"code": "output_too_large", "message": "result exceeds bound"}})
        self.e.store.event(rid, "tool:" + name, not code, code)
        return {"content": [{"type": "text", "text": text}], "isError": bool(code)}

    @staticmethod
    def read_line(stream) -> Tuple[Optional[bytes], bool]:
        chunks, size, over = [], 0, False
        while True:
            part = stream.readline(Hard.MAX_LINE_BYTES + 1)
            if not part:
                return (None, False) if not chunks and not over else (b"".join(chunks), over)
            size += len(part)
            if size > Hard.MAX_LINE_BYTES:
                over, chunks = True, []
            elif not over:
                chunks.append(part)
            if part.endswith(b"\n"):
                return b"".join(chunks), over

    def serve(self, stdin=None, stdout=None) -> int:
        stdin, out = stdin or sys.stdin.buffer, stdout or sys.stdout.buffer
        sys.stdout = sys.stderr  # stray prints can never corrupt the protocol stream
        LOG.info("MCP server ready (v%s)", VERSION)
        while True:
            line, over = self.read_line(stdin)
            if line is None:
                break
            if over:
                resp = self._err(None, -32600, "Request too large")
            else:
                line = line.strip()
                if not line:
                    continue
                resp = self.handle_bytes(line)
            if resp is not None:
                out.write(json.dumps(resp, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
                out.flush()
        return 0


# =====================================================================================
# PLUGIN BUILDER + INSTALLER
# =====================================================================================
class PluginBuilder:
    """Generates ONE LM Studio plugin project (runtime artifact, not a project source file).
    It shells out to nothing: the TypeScript calls `python nythosplus.py advise --stdin --json` via execFile (no shell).
    NOT VERIFIED LIVE: written against the public @lmstudio/sdk plugin API; run `lms dev` in the folder to load it."""

    def __init__(self, python: str, script: str, install_id: str):
        self.py, self.script, self.iid = python, script, install_id

    def files(self) -> Dict[str, str]:
        P, S = json.dumps(self.py), json.dumps(self.script)
        hdr = "// Generated by nythosplus.py (owner marker: %s). Do not edit; regenerate with `python nythosplus.py install`.\n" % (OWNER_PREFIX + self.iid)
        return {
            "manifest.json": json.dumps({"type": "plugin", "runner": "node", "owner": "nythosplus", "name": "nythosplus", "revision": 1}, indent=2) + "\n",
            "package.json": json.dumps({"name": "nythosplus", "version": VERSION, "private": True, "scripts": {"dev": "lms dev"},
                                        "dependencies": {"@lmstudio/sdk": "latest", "zod": "^3.23.0"}, "devDependencies": {"typescript": "^5.4.0"}}, indent=2) + "\n",
            "tsconfig.json": json.dumps({"compilerOptions": {"target": "ES2022", "module": "commonjs", "strict": True, "esModuleInterop": True, "outDir": "dist"}, "include": ["src"]}, indent=2) + "\n",
            "src/advise.ts": hdr + f"""import {{ execFile }} from "child_process";
const PYTHON = {P};
const SCRIPT = {S};
export function advise(prompt: string, model: string): Promise<any> {{
  return new Promise((resolve) => {{
    const child = execFile(PYTHON, [SCRIPT, "advise", "--stdin", "--json"], {{ timeout: 5000, maxBuffer: 1 << 20, shell: false }}, (err, stdout) => {{
      if (err) return resolve(null);
      try {{ resolve(JSON.parse(stdout)); }} catch {{ resolve(null); }}
    }});
    child.stdin?.end(JSON.stringify({{ prompt: prompt.slice(0, 8000), model }}));
  }});
}}
""",
            "src/config.ts": hdr + """import { createConfigSchematics } from "@lmstudio/sdk";
export const configSchematics = createConfigSchematics()
  .field("enabled", "boolean", { displayName: "Adaptive effort (AUTO)" }, true)
  .field("injectHints", "boolean", { displayName: "Inject effort hints (models without native control)" }, true)
  .build();
""",
            "src/toolsProvider.ts": hdr + """import { tool, type ToolsProviderController } from "@lmstudio/sdk";
import { z } from "zod";
import { advise } from "./advise";
export const toolsProvider: ToolsProviderController = async () => [
  tool({
    name: "nythosplus_effort_advice",
    description: "Recommend a minimum sufficient reasoning effort (E0-E4) for a prompt. Advice only; changes no model settings.",
    parameters: { prompt: z.string().max(8000), model: z.string().max(128).optional() },
    implementation: async ({ prompt, model }) => (await advise(prompt, model ?? "unknown")) ?? { error: "NOT_AVAILABLE" },
  }),
];
""",
            "src/promptPreprocessor.ts": hdr + """import { type PromptPreprocessor } from "@lmstudio/sdk";
import { configSchematics } from "./config";
import { advise } from "./advise";
export const promptPreprocessor: PromptPreprocessor = async (ctl, userMessage) => {
  const cfg = ctl.getPluginConfig(configSchematics);
  if (!cfg.get("enabled") || !cfg.get("injectHints")) return userMessage;
  const text = userMessage.getText();
  const adv = await advise(text, "unknown");
  const hint = adv?.hint;
  return hint ? `${hint}\\n\\n${text}` : userMessage;
};
""",
            "src/index.ts": hdr + """import { type PluginContext } from "@lmstudio/sdk";
import { configSchematics } from "./config";
import { toolsProvider } from "./toolsProvider";
import { promptPreprocessor } from "./promptPreprocessor";
export async function main(context: PluginContext) {
  context.withConfigSchematics(configSchematics);
  context.withToolsProvider(toolsProvider);
  context.withPromptPreprocessor(promptPreprocessor);
}
""",
        }

    @staticmethod
    def validate(files: Dict[str, str]) -> List[str]:
        problems = []
        for need in ("manifest.json", "package.json", "src/index.ts"):
            if need not in files:
                problems.append(f"missing {need}")
        for n in ("manifest.json", "package.json", "tsconfig.json"):
            with contextlib.suppress(KeyError):
                try:
                    json.loads(files[n])
                except ValueError:
                    problems.append(f"{n} is not valid JSON")
        for n, t in files.items():
            if n.endswith(".ts"):
                if t.count("{") != t.count("}") or t.count("(") != t.count(")"):
                    problems.append(f"{n}: unbalanced brackets")
                if re.search(r"shell:\s*true|child_process\.exec\(|\beval\(", t):
                    problems.append(f"{n}: forbidden construct")
            if ".." in n or n.startswith(("/", "\\")):
                problems.append(f"{n}: unsafe path")
        return problems


class Report:
    def __init__(self, title: str):
        self.title, self.rows, self.failed = title, [], False

    def add(self, level: str, name: str, msg: str) -> None:
        self.rows.append((level, name, msg))
        self.failed = self.failed or level == "FAIL"

    ok = lambda s, n, m: s.add("PASS", n, m)
    warn = lambda s, n, m: s.add("WARN", n, m)
    fail = lambda s, n, m: s.add("FAIL", n, m)
    info = lambda s, n, m: s.add("INFO", n, m)
    skip = lambda s, n, m: s.add("SKIP", n, m)


def _strict_json(raw: bytes) -> Any:
    def hook(pairs):
        seen = set()
        for k, _ in pairs:
            if k in seen:
                raise PlusError("invalid_json", f"duplicate key '{k}'")
            seen.add(k)
        return dict(pairs)
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=hook)
    except PlusError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError) as e:
        raise PlusError("invalid_json", str(e)[:100])


class Installer:
    def __init__(self, paths: Paths, config_override: Optional[str] = None):
        self.paths, self.override = paths, config_override

    def locate(self) -> Tuple[Optional[Path], str]:
        if self.override:
            p = Path(self.override)
            return (p, "") if p.name == "mcp.json" else (None, "--config must point to a file named mcp.json")
        d = Path(os.environ.get("NYTHOSPLUS_LMSTUDIO_DIR") or (Path.home() / ".lmstudio"))
        return (d / "mcp.json", "") if d.is_dir() else (None, f"LM Studio configuration directory not found ({d}); start LM Studio once or pass --config")

    def _install_id(self) -> str:
        path, _ = self.locate()
        if path is not None and path.exists():
            with contextlib.suppress(PlusError, OSError, KeyError, TypeError, ValueError):
                data, _raw = self._load(path)
                cur = data.get("mcpServers", {}).get(SERVER_KEY)
                if self.is_owned(cur):
                    i = cur["env"][OWNER_ENV_KEY][len(OWNER_PREFIX):]
                    if re.fullmatch(r"[0-9a-f]{32}", i):
                        return i
        st, _ = read_state(self.paths)
        i = (st.get("install") or {}).get("install_id") if isinstance(st.get("install"), dict) else None
        return i if isinstance(i, str) and re.fullmatch(r"[0-9a-f]{32}", i) else uuid.uuid4().hex

    def desired_entry(self, iid: str) -> Dict[str, Any]:
        env = {OWNER_ENV_KEY: OWNER_PREFIX + iid}
        if os.environ.get("NYTHOSPLUS_HOME"):
            env["NYTHOSPLUS_HOME"] = os.environ["NYTHOSPLUS_HOME"]
        return {"command": sys.executable, "args": [str(Paths.script()), "--mcp"], "env": env}

    @staticmethod
    def is_owned(e: Any) -> bool:
        return (isinstance(e, dict) and isinstance(e.get("env"), dict) and str(e["env"].get(OWNER_ENV_KEY, "")).startswith(OWNER_PREFIX)
                and isinstance(e.get("args"), list) and "--mcp" in e["args"])

    def backup(self, src: Path, raw: bytes, reason: str) -> Dict[str, Any]:
        self.paths.ensure()
        dest = self.paths.backups / f"mcp.json.{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.bak"
        atomic_write(dest, raw)
        dg = sha256_bytes(raw)
        if sha256_bytes(dest.read_bytes()) != dg:
            raise PlusError("backup_failed", "backup verification failed")
        meta = {"time": iso(now()), "source": str(src), "backup": str(dest), "sha256": dg, "reason": reason}
        try:
            idx = json.loads(self.paths.backup_index.read_text("utf-8")) if self.paths.backup_index.exists() else []
            idx = idx if isinstance(idx, list) else []
        except (OSError, ValueError):
            idx = []
        atomic_write(self.paths.backup_index, json.dumps((idx + [meta])[-200:], indent=2))
        return meta

    def _load(self, path: Path) -> Tuple[Dict[str, Any], Optional[bytes]]:
        if not path.exists():
            return {}, None
        raw = path.read_bytes()
        d = _strict_json(raw)
        if not isinstance(d, dict) or ("mcpServers" in d and not isinstance(d["mcpServers"], dict)):
            raise PlusError("invalid_json", "mcp.json must be an object with an object 'mcpServers'")
        return d, raw

    def plugin_status(self) -> Dict[str, Any]:
        d = self.paths.plugin_dir
        m = d / PLUGIN_MARKER
        if not m.exists():
            return {"present": False, "owned": False, "intact": False, "path": str(d)}
        try:
            meta = json.loads(m.read_text("utf-8"))
            intact = all((d / n).exists() and sha256_bytes((d / n).read_bytes()) == h for n, h in meta["files"].items())
            return {"present": True, "owned": True, "intact": intact, "path": str(d)}
        except (OSError, ValueError, KeyError):
            return {"present": True, "owned": True, "intact": False, "path": str(d)}

    def _record(self, iid: str, path: Optional[Path]) -> None:
        st, _ = read_state(self.paths)
        st["install"] = {"install_id": iid, "config_path": str(path) if path else None, "updated_at": iso(now())}
        write_state(self.paths, st)

    def _write_plugin(self, iid: str, r: Report) -> None:
        pb = PluginBuilder(sys.executable, str(Paths.script()), iid)
        files = pb.files()
        probs = PluginBuilder.validate(files)
        if probs:
            r.fail("plugin", "generated plugin failed validation: " + "; ".join(probs))
            return
        d = self.paths.plugin_dir
        if d.exists() and not (d / PLUGIN_MARKER).exists() and any(d.iterdir()):
            r.fail("plugin", f"{d} exists and is not Nythos Plus-owned; refusing to write")
            return
        written: List[Path] = []
        try:
            for n, t in files.items():
                p = SecurityGuard.resolve_inside(d, n)
                atomic_write(p, t)
                written.append(p)
                if sha256_bytes(p.read_bytes()) != sha256_bytes(t.encode("utf-8")):
                    raise PlusError("verify_failed", f"{n} read-back mismatch")
            atomic_write(d / PLUGIN_MARKER, json.dumps({"owner": OWNER_PREFIX + iid, "files": {n: sha256_bytes(t.encode()) for n, t in files.items()}}, indent=2))
            r.ok("plugin", f"generated {len(files)} files in {d}")
            r.skip("plugin_live", "NOT VERIFIED LIVE: load with `lms dev` inside that folder (needs Node + lms CLI)")
        except (PlusError, OSError) as e:
            for p in written:
                with contextlib.suppress(OSError):
                    p.unlink()
            r.fail("plugin", f"{getattr(e, 'message', str(e))}; rolled back generated files")

    def install(self, dry_run: bool = False, mcp: bool = True, plugin: bool = True) -> Report:
        r = Report("install")
        r.ok("platform", "Windows") if is_windows() else r.warn("platform", f"not Windows ({sys.platform}); Windows-first tool, continuing")
        if sys.version_info < (3, 9):
            r.fail("python", "Python 3.9+ required")
            return r
        r.ok("python", f"{sys.version.split()[0]} at {sys.executable}")
        iid = self._install_id()
        if plugin:
            if dry_run:
                r.info("plugin", f"would generate plugin files in {self.paths.plugin_dir}")
            else:
                self.paths.ensure()
                self._write_plugin(iid, r)
        if not mcp:
            return r
        path, why = self.locate()
        if not path:
            (r.warn if dry_run else r.fail)("mcp", why + " (MCP registration NOT done)")
            return r
        try:
            data, raw = self._load(path)
        except PlusError as e:
            r.fail("config_json", f"mcp.json invalid - NOT modified: {e.message}")
            return r
        servers = data.get("mcpServers", {})
        cur = servers.get(SERVER_KEY)
        if cur is not None and not self.is_owned(cur):
            r.fail("ownership", f"entry '{SERVER_KEY}' exists but is not Nythos Plus-owned; refusing to overwrite")
            return r
        if any(k != SERVER_KEY and self.is_owned(v) for k, v in servers.items()):
            r.fail("duplicates", "Nythos Plus-owned entry under another name; resolve manually")
            return r
        if cur is not None:
            iid = cur["env"][OWNER_ENV_KEY][len(OWNER_PREFIX):]
        desired = self.desired_entry(iid)
        if cur == desired:
            r.ok("registration", "already registered and up to date; mcp.json not modified")
            if not dry_run:
                with contextlib.suppress(OSError, PlusError):
                    self._record(iid, path)
            return r
        if dry_run:
            r.info("mcp", "would add/update only the nythosplus entry; nothing written")
            return r
        existed = raw is not None
        try:
            if existed:
                meta = self.backup(path, raw, "install")
                r.ok("backup", f"{meta['backup']} (sha256 {meta['sha256'][:16]}...)")
            new = copy.deepcopy(data)
            new.setdefault("mcpServers", {})[SERVER_KEY] = desired
            atomic_write(path, json.dumps(new, indent=2, ensure_ascii=False) + "\n")
            d2, _ = self._load(path)
            owned = [k for k, v in d2.get("mcpServers", {}).items() if self.is_owned(v)]
            if d2 != new or owned != [SERVER_KEY]:
                raise PlusError("verify_failed", "written config differs or entry not registered exactly once")
            if {k: v for k, v in d2.items() if k != "mcpServers"} != {k: v for k, v in data.items() if k != "mcpServers"} or \
                    {k: v for k, v in d2["mcpServers"].items() if k != SERVER_KEY} != {k: v for k, v in servers.items() if k != SERVER_KEY}:
                raise PlusError("verify_failed", "unrelated configuration changed")
            r.ok("verify", "re-read OK; registered exactly once; all unrelated entries preserved")
        except (PlusError, OSError) as e:
            r.fail("verify", f"{getattr(e, 'message', str(e))}; rolling back")
            with contextlib.suppress(OSError):
                if existed:
                    atomic_write(path, raw)
                    r.info("rollback", "original mcp.json restored")
                elif path.exists():
                    path.unlink()
            return r
        with contextlib.suppress(OSError, PlusError):
            self._record(iid, path)
        r.skip("live_check", "NOT VERIFIED LIVE: LM Studio launching Nythos Plus was not tested by the installer")
        return r

    def uninstall(self, dry_run: bool = False) -> Report:
        r = Report("uninstall")
        d = self.paths.plugin_dir
        if (d / PLUGIN_MARKER).exists():
            if dry_run:
                r.info("plugin", "would remove generated plugin files")
            else:
                try:
                    meta = json.loads((d / PLUGIN_MARKER).read_text("utf-8"))
                    for n in meta["files"]:
                        with contextlib.suppress(OSError):
                            SecurityGuard.resolve_inside(d, n).unlink()
                    (d / PLUGIN_MARKER).unlink()
                    for sub in sorted(d.rglob("*"), reverse=True):
                        with contextlib.suppress(OSError):
                            sub.rmdir()
                    with contextlib.suppress(OSError):
                        d.rmdir()
                    r.ok("plugin", "generated plugin files removed (only files listed in the ownership marker)")
                except (OSError, ValueError, KeyError, PlusError) as e:
                    r.warn("plugin", f"could not fully remove plugin: {e}")
        path, why = self.locate()
        if not path or not path.exists():
            r.ok("registration", "no mcp.json; nothing to unregister")
            return r
        try:
            data, raw = self._load(path)
        except PlusError as e:
            r.fail("config_json", f"mcp.json invalid - NOT modified: {e.message}")
            return r
        cur = data.get("mcpServers", {}).get(SERVER_KEY)
        if cur is None:
            r.ok("registration", "not registered; nothing to remove")
            return r
        if not self.is_owned(cur):
            r.fail("ownership", f"entry '{SERVER_KEY}' is not Nythos Plus-owned; refusing to remove it")
            return r
        if dry_run:
            r.info("mcp", "would remove only the nythosplus entry")
            return r
        try:
            meta = self.backup(path, raw, "uninstall")
            r.ok("backup", f"{meta['backup']} (sha256 {meta['sha256'][:16]}...)")
            new = copy.deepcopy(data)
            del new["mcpServers"][SERVER_KEY]
            atomic_write(path, json.dumps(new, indent=2, ensure_ascii=False) + "\n")
            d2, _ = self._load(path)
            if d2 != new:
                raise PlusError("verify_failed", "post-write verification failed")
            r.ok("verify", "nythosplus entry removed; all other entries preserved")
        except (PlusError, OSError) as e:
            r.fail("verify", f"{getattr(e, 'message', str(e))}; restoring original")
            with contextlib.suppress(OSError):
                atomic_write(path, raw)
        r.info("data", f"user data kept at {self.paths.home}")
        return r

    def status(self) -> Dict[str, Any]:
        path, why = self.locate()
        s = {"config_path": str(path) if path else None, "exists": False, "valid_json": None, "registered": False, "owned": False,
             "matches": False, "problem": why, "plugin": self.plugin_status()}
        if not path or not path.exists():
            s["exists"] = bool(path and path.exists())
            return s
        s["exists"] = True
        try:
            data, _ = self._load(path)
        except PlusError as e:
            s.update(valid_json=False, problem=e.message)
            return s
        s["valid_json"] = True
        cur = data.get("mcpServers", {}).get(SERVER_KEY)
        s["registered"], s["owned"] = cur is not None, self.is_owned(cur)
        if s["owned"]:
            s["matches"] = cur == self.desired_entry(cur["env"][OWNER_ENV_KEY][len(OWNER_PREFIX):])
        return s

    def repair(self) -> Report:
        r = Report("repair")
        try:
            eng = Engine(self.paths)
            ok, msg = eng.store.integrity()
            (r.ok if ok else r.fail)("database", f"schema v{eng.store.schema_version()}, integrity: {msg}")
        except PlusError as e:
            r.fail("database", e.message)
        n = clean_stale_tmp(self.paths.home) + clean_stale_tmp(self.paths.backups)
        path, _ = self.locate()
        if path:
            n += clean_stale_tmp(path.parent)
        r.ok("temp_files", f"removed {n} stale temp file(s)")
        st = self.status()
        if st["valid_json"] is False:
            r.fail("config_json", f"mcp.json invalid ({st['problem']}); not auto-modified. Backups: {self.paths.backups}")
        elif st["owned"] and not st["matches"] or (st["plugin"]["present"] and not st["plugin"]["intact"]):
            sub = self.install()
            r.rows.extend(sub.rows)
            r.failed = r.failed or sub.failed
        elif st["registered"] and not st["owned"]:
            r.warn("registration", "entry 'nythosplus' is not Nythos Plus-owned; left untouched")
        else:
            r.info("registration", "up to date" if st["owned"] else "not installed; run: python nythosplus.py install")
        return r


# =====================================================================================
# CLI STYLE / LOGO
# =====================================================================================
LOGO_GRID = ("..########..", "..#E####E#..", "############", "############", "..########..", "..########..", "..#.#..#.#..", "..#.#..#.#..")
LOGO_PALETTE = {"#": (217, 119, 87), "E": (0, 0, 0)}
ORANGE, CREAM, DIM, GREEN, YELLOW, RED = (217, 119, 87), (240, 230, 210), (140, 130, 118), (120, 200, 120), (230, 190, 80), (225, 85, 85)


def enable_ansi() -> bool:
    if not is_windows():
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h, m = k.GetStdHandle(-11), ctypes.c_uint32()
        return bool(k.GetConsoleMode(h, ctypes.byref(m)) and k.SetConsoleMode(h, m.value | 4))
    except Exception:
        return False


class Style:
    def __init__(self, color: Optional[bool] = None):
        if color is None:
            color = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb" and enable_ansi()
        self.color = bool(color)

    def fg(self, t: str, rgb, bold: bool = False) -> str:
        return f"\x1b[{'1;' if bold else ''}38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{t}\x1b[0m" if self.color else t

    def bg(self, t: str, rgb) -> str:
        return f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m{t}\x1b[0m" if self.color else t


def render_logo(style: Optional[Style] = None) -> str:
    style = style or Style()
    lines = []
    for row in LOGO_GRID:
        cells = []
        for ch in row:
            rgb = LOGO_PALETTE.get(ch)
            cells.append("  " if rgb is None else style.bg("  ", rgb) if style.color else ("  " if ch == "E" else "##"))
        lines.append("  " + "".join(cells))
    return "\n".join(lines)


def _lvl(level: str):
    return {"PASS": GREEN, "WARN": YELLOW, "FAIL": RED, "SKIP": DIM}.get(level, CREAM)


def print_report(rep: Report, st: Style) -> None:
    print(st.fg(f"\n{APP} {rep.title.upper()}", ORANGE, True))
    for level, name, msg in rep.rows:
        print(f"  {st.fg(level.ljust(4), _lvl(level), True)}  {name.ljust(18)} {msg}")
    print()


# =====================================================================================
# DIAGNOSTICS
# =====================================================================================
def mcp_probe(timeout: float = 30.0) -> Tuple[bool, str]:
    tmp = tempfile.mkdtemp(prefix="nythosplus-probe-")
    try:
        env = dict(os.environ, NYTHOSPLUS_HOME=tmp, PYTHONIOENCODING="utf-8")
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "nythosplus_status", "arguments": {}}}]
        payload = b"".join(json.dumps(m).encode() + b"\n" for m in msgs) + b"{bad\n" + json.dumps({"jsonrpc": "2.0", "id": 4, "method": "ping"}).encode() + b"\n"
        p = subprocess.run([sys.executable, str(Paths.script()), "--mcp"], input=payload, capture_output=True, timeout=timeout, env=env)
        parsed = []
        for ln in p.stdout.split(b"\n"):
            if ln.strip():
                try:
                    parsed.append(json.loads(ln))
                except ValueError:
                    return False, "stdout contained non-JSON output"
        ids = [m.get("id") for m in parsed]
        if p.returncode != 0 or ids != [1, 2, 3, None, 4]:
            return False, f"exit={p.returncode} ids={ids}"
        if len(parsed[1]["result"]["tools"]) != 5 or parsed[2]["result"].get("isError"):
            return False, "tool list/status call unexpected"
        return True, "initialize / tools.list / tools.call / ping OK; malformed line rejected; stdout is protocol-only"
    except subprocess.TimeoutExpired:
        return False, "probe timed out"
    except Exception as e:
        return False, f"probe failed: {type(e).__name__}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class Diagnostics:
    def __init__(self, paths: Paths, installer: Installer):
        self.paths, self.inst = paths, installer

    def doctor(self) -> Report:
        r = Report("doctor")
        v = sys.version_info
        (r.ok if v >= (3, 9) else r.fail)("python", f"{v.major}.{v.minor}.{v.micro}")
        (r.ok if is_windows() else r.warn)("windows", "Windows" if is_windows() else f"running on {sys.platform} (Windows-first tool)")
        try:
            self.paths.ensure()
            probe = self.paths.home / f".w-{uuid.uuid4().hex[:6]}"
            atomic_write(probe, "x")
            probe.unlink()
            r.ok("directory", f"{self.paths.home} (writable, atomic write OK)")
        except OSError as e:
            r.fail("directory", f"not writable: {e}")
            return r
        eng = None
        try:
            eng = Engine(self.paths)
            r.ok("database", f"opened; journal_mode={eng.store.journal_mode()}")
            ok, msg = eng.store.integrity()
            (r.ok if ok else r.fail)("integrity", msg)
            sv = eng.store.schema_version()
            (r.ok if sv == SCHEMA_VERSION else r.fail)("schema", f"version {sv} (expected {SCHEMA_VERSION})")
            (r.warn if eng.cfg.warning else r.ok)("settings", eng.cfg.warning or "config valid (hysteresis thresholds distinct)")
        except PlusError as e:
            r.fail("database", e.message)
        rg = ResourceGuard().sample(force=True)
        (r.ok if rg.observed else r.warn)("resources", f"{rg.public()}" if rg.observed else "no resource data observable (NOT_AVAILABLE)")
        cfg = eng.cfg if eng else Config()
        try:
            br = LMStudioBridge(cfg.base_url, 3.0)
            flavor, rows = br.list_models()
            if flavor == "unavailable":
                r.skip("lm_studio_api", f"NOT_AVAILABLE: {cfg.base_url} not reachable (start LM Studio's local server)")
            else:
                loaded = [m["id"] for m in rows if m.get("state") == "loaded"]
                r.ok("lm_studio_api", f"{flavor}: {len(rows)} models, loaded: {loaded if flavor == 'native_v0' else 'state unknown (/v1/models only)'}")
                if flavor != "native_v0":
                    r.warn("capabilities", "/api/v0/models missing: load state cannot be verified, so runs will be refused (fail closed)")
        except PlusError as e:
            r.fail("lm_studio_api", e.message)
        lms = shutil.which("lms")
        if lms:
            try:
                p = subprocess.run([lms, "version"], capture_output=True, text=True, timeout=5)
                r.ok("lms_cli", (p.stdout or p.stderr).strip().splitlines()[0][:80] if (p.stdout or p.stderr).strip() else "present")
            except Exception:
                r.warn("lms_cli", "present but did not answer")
        else:
            r.skip("lms_cli", "lms CLI not found on PATH")
        st = self.inst.status()
        if st["config_path"] is None:
            r.warn("mcp", st["problem"])
        elif not st["exists"]:
            r.warn("mcp", f"{st['config_path']} does not exist (run install)")
        elif st["valid_json"] is False:
            r.fail("mcp", f"mcp.json invalid: {st['problem']}")
        elif not st["registered"]:
            r.warn("mcp", "not registered (run install)")
        else:
            (r.ok if st["owned"] and st["matches"] else r.warn)("mcp", "registered, owned, up to date" if st["owned"] and st["matches"] else "registered but not owned/up to date")
        pl = st["plugin"]
        (r.ok if pl["intact"] else r.skip if not pl["present"] else r.warn)("plugin", "generated files intact" if pl["intact"] else ("not generated" if not pl["present"] else "files modified (run repair)"))
        for bad in ("http://example.com:80", "http://192.168.0.5:1234", "https://127.0.0.1:1234"):
            try:
                SecurityGuard.local_url(bad)
                r.fail("security", f"non-loopback URL accepted: {bad}")
                break
            except PlusError:
                pass
        else:
            r.ok("security", "non-loopback / non-http URLs rejected")
        ok, msg = mcp_probe()
        (r.ok if ok else r.fail)("runtime_protocol", msg)
        r.skip("lm_studio_live", "NOT VERIFIED LIVE: doctor performs no inference")
        return r


# =====================================================================================
# CLI (single entrypoint; every command is wired to the classes above - no second runtime)
# =====================================================================================
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_INTERNAL, EXIT_INTERRUPT = 0, 1, 2, 70, 130
ADVISE_MAX_STDIN = 262_144
CLI_COMMANDS = ("status", "doctor", "self-test", "install", "repair", "uninstall", "models", "policy", "benchmark", "history", "advise")
USAGE_CODES = {"invalid_argument", "unknown_argument", "too_long", "out_of_range", "invalid_id", "invalid_json", "too_large"}


def _setup_logging(verbose: bool = False) -> None:
    """All logging goes to stderr. stdout is reserved for command output / the MCP protocol stream."""
    for h in list(LOG.handlers):
        LOG.removeHandler(h)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s nythosplus: %(message)s"))
    LOG.addHandler(h)
    LOG.propagate = False
    LOG.setLevel(logging.INFO if verbose else logging.WARNING)


def _emit_json(obj: Any, compact: bool = False) -> None:
    kw: Dict[str, Any] = {"separators": (",", ":")} if compact else {"indent": 2}
    sys.stdout.write(json.dumps(obj, ensure_ascii=True, **kw) + "\n")
    sys.stdout.flush()


def _say(text: str = "") -> None:
    sys.stdout.write(text + "\n")


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return NA
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _paths(a: argparse.Namespace) -> Paths:
    home = getattr(a, "home", None)
    if home:
        os.environ["NYTHOSPLUS_HOME"] = str(Path(home).expanduser().resolve())  # restored by main(); keeps registered entry consistent
    return Paths()


def _installer(a: argparse.Namespace, paths: Paths) -> Installer:
    return Installer(paths, getattr(a, "config", None))


def _report_obj(rep: Report) -> Dict[str, Any]:
    return {"title": rep.title, "ok": not rep.failed, "steps": [{"level": lv, "step": n, "message": m} for lv, n, m in rep.rows]}


def _finish(rep: Report, a: argparse.Namespace) -> int:
    if getattr(a, "json", False):
        _emit_json(_report_obj(rep))
    else:
        print_report(rep, Style())
    return EXIT_FAIL if rep.failed else EXIT_OK


def _engine_if_present(paths: Paths) -> Optional[Engine]:
    return Engine(paths) if paths.db.exists() else None


def _db_summary(paths: Paths) -> Dict[str, Any]:
    base: Dict[str, Any] = {"path": str(paths.db), "initialized": paths.db.exists()}
    if not base["initialized"]:
        return base
    st = Store(paths.db)
    try:
        ok, msg = st.integrity()
        counts: Dict[str, int] = {}
        with st.read() as c:
            for t in ("models", "decisions", "outcomes", "attempts", "events"):
                counts[t] = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        base.update(schema_version=st.schema_version(), integrity_ok=ok, integrity=msg, counts=counts)
    except (sqlite3.Error, PlusError) as e:
        base.update(integrity_ok=False, integrity=f"unreadable: {str(getattr(e, 'message', e))[:100]}")
    return base


def _mcp_desc(st: Dict[str, Any]) -> str:
    if st["config_path"] is None:
        return f"mcp.json not located ({st['problem']})"
    if not st["exists"]:
        return f"{st['config_path']} does not exist"
    if st["valid_json"] is False:
        return f"{st['config_path']} is INVALID: {st['problem']}"
    if not st["registered"]:
        return f"{st['config_path']}: not registered"
    if not st["owned"]:
        return f"{st['config_path']}: 'nythosplus' entry exists but is NOT Nythos Plus-owned (left untouched)"
    return f"{st['config_path']}: registered, owned, " + ("up to date" if st["matches"] else "needs repair")


# ---- commands -----------------------------------------------------------------------
def cmd_status(a: argparse.Namespace) -> int:
    paths = _paths(a)
    inst = _installer(a, paths)
    cfg = Config.load(paths)
    out: Dict[str, Any] = {"app": APP, "version": VERSION, "python": sys.version.split()[0], "data_dir": str(paths.home),
                           "database": _db_summary(paths), "registration": inst.status(), "settings_warning": cfg.warning or None}
    if getattr(a, "no_lm_studio", False):
        out["lm_studio"] = {"checked": False}
    else:
        try:
            flavor, rows = LMStudioBridge(cfg.base_url, 3.0).list_models()
            out["lm_studio"] = {"checked": True, "base_url": cfg.base_url, "api": flavor, "reachable": flavor != "unavailable",
                                "load_state_verifiable": flavor == "native_v0",
                                "loaded_models": [r["id"] for r in rows if r.get("state") == "loaded"] if flavor == "native_v0" else None}
        except PlusError as e:
            out["lm_studio"] = {"checked": True, "error": e.message}
    out["live_integration"] = "NOT VERIFIED LIVE (status does not launch the plugin or MCP server from LM Studio)"
    bad = out["database"].get("integrity_ok") is False or out["registration"]["valid_json"] is False
    if getattr(a, "json", False):
        _emit_json(out)
        return EXIT_FAIL if bad else EXIT_OK
    db, st, lm = out["database"], out["registration"], out["lm_studio"]
    _say(f"{APP} {VERSION} - {TAGLINE}")
    _say(f"  data dir     : {paths.home}")
    if db["initialized"]:
        cnt = db.get("counts", {})
        _say(f"  database     : schema v{db.get('schema_version', '?')}, integrity {db.get('integrity')}, outcomes={cnt.get('outcomes', '?')}, models={cnt.get('models', '?')}")
    else:
        _say("  database     : not initialized (run: python nythosplus.py install)")
    _say(f"  mcp.json     : {_mcp_desc(st)}")
    pl = st["plugin"]
    _say("  plugin       : " + ("generated, intact" if pl["intact"] else "generated, MODIFIED (run repair)" if pl["present"] else "not generated"))
    if not lm.get("checked"):
        _say("  lm studio    : not checked (--no-lm-studio)")
    elif "error" in lm:
        _say(f"  lm studio    : {lm['error']}")
    elif not lm["reachable"]:
        _say(f"  lm studio    : {cfg.base_url} not reachable (NOT_AVAILABLE)")
    else:
        _say(f"  lm studio    : {lm['base_url']} api={lm['api']} loaded={lm['loaded_models'] if lm['load_state_verifiable'] else 'unknown (runs fail closed)'}")
    _say(f"  live check   : {out['live_integration']}")
    if cfg.warning:
        _say(f"  settings     : WARNING {cfg.warning}")
    return EXIT_FAIL if bad else EXIT_OK


def cmd_doctor(a: argparse.Namespace) -> int:
    paths = _paths(a)
    return _finish(Diagnostics(paths, _installer(a, paths)).doctor(), a)


def cmd_self_test(a: argparse.Namespace) -> int:
    rep = SelfTest().run()
    _setup_logging(bool(getattr(a, "verbose", False)))
    return _finish(rep, a)


def cmd_install(a: argparse.Namespace) -> int:
    paths = _paths(a)
    inst = _installer(a, paths)
    rep = Report("install")
    if a.dry_run:
        rep.info("database", f"would initialize state and database under {paths.home}")
    else:
        try:
            eng = Engine(paths)
            ok, msg = eng.store.integrity()
            (rep.ok if ok else rep.fail)("database", f"{paths.db} ready (schema v{eng.store.schema_version()}, integrity {msg})")
            if not paths.config.exists():
                write_state(paths, {"settings": {}})
                rep.ok("state", f"created {paths.config}")
            else:
                rep.ok("state", "existing config.json kept")
            if eng.cfg.warning:
                rep.warn("settings", eng.cfg.warning)
        except (PlusError, OSError, sqlite3.Error) as e:
            rep.fail("database", f"{getattr(e, 'message', str(e))}; nothing else was changed")
        if rep.failed:
            return _finish(rep, a)
    sub = inst.install(dry_run=a.dry_run, mcp=not a.skip_mcp, plugin=not a.skip_plugin)
    rep.rows.extend(sub.rows)
    rep.failed = rep.failed or sub.failed
    if a.dry_run:
        rep.info("dry_run", "no file was written or modified")
    return _finish(rep, a)


def cmd_repair(a: argparse.Namespace) -> int:
    paths = _paths(a)
    inst = _installer(a, paths)
    rep = inst.repair()
    st, _ = read_state(paths)
    if isinstance(st.get("install"), dict) and not inst.plugin_status()["present"]:
        sub = inst.install(mcp=False)  # plugin artifacts only; registration already handled above
        rep.add("INFO", "plugin_regen", "plugin was missing after a previous install; regenerating Nythos Plus-owned files only")
        rep.rows.extend(sub.rows)
        rep.failed = rep.failed or sub.failed
    n = sum(clean_stale_tmp(d) for d in (paths.plugin_dir, paths.plugin_dir / "src") if d.is_dir())
    rep.ok("plugin_tmp", f"removed {n} stale temp file(s) in the plugin folder")
    pl = inst.plugin_status()
    if pl["present"]:
        (rep.ok if pl["intact"] else rep.fail)("plugin_check", "generated plugin files intact" if pl["intact"] else "plugin files still differ from the ownership marker")
    return _finish(rep, a)


def cmd_uninstall(a: argparse.Namespace) -> int:
    if a.purge_data and not a.yes:
        raise PlusError("invalid_argument", "--purge-data is destructive and requires --yes")
    paths = _paths(a)
    inst = _installer(a, paths)
    rep = inst.uninstall(dry_run=a.dry_run)
    if not a.dry_run and not rep.failed:
        st = inst.status()
        (rep.fail if st["owned"] else rep.ok)("post_check", "Nythos Plus entry still present" if st["owned"] else "no Nythos Plus-owned registration remains")
    if a.purge_data:
        if rep.failed:
            rep.warn("purge", "skipped because uninstall reported failures")
        elif a.dry_run:
            rep.info("purge", "would delete the database, its WAL/SHM files and config.json (backups are kept)")
        else:
            gone = []
            for p in (paths.db, Path(str(paths.db) + "-wal"), Path(str(paths.db) + "-shm"), paths.config):
                if p.exists():
                    p.unlink()
                    gone.append(p.name)
            rep.ok("purge", f"deleted {', '.join(gone) or 'nothing'}; backups kept in {paths.backups}")
    return _finish(rep, a)


def cmd_models(a: argparse.Namespace) -> int:
    paths = _paths(a)
    cfg = Config.load(paths)
    if a.probe:
        SecurityGuard.ident(a.probe, "model")
        vals = [v.strip() for v in a.values.split(",")] if a.values else None
        if vals is not None and not (2 <= len(vals) <= 6 and all(re.fullmatch(r"[a-z]{3,10}", v) for v in vals)):
            raise PlusError("invalid_argument", "--values must be 2-6 lowercase words, e.g. low,medium,high")
        eng = Engine(paths)
        res = probe_native_effort(LMStudioBridge(cfg.base_url, cfg.http_timeout), a.probe, vals)  # fails closed if not loaded
        eng.profiles.upsert(a.probe, native_state=res["state"], native_values=res["values"])
        if a.json:
            _emit_json({"model": a.probe, **res})
        else:
            _say(f"{a.probe}: native effort {res['state']} - {res['detail']}")
        return EXIT_OK
    flavor, rows = LMStudioBridge(cfg.base_url, 3.0).list_models()
    if flavor == "unavailable":
        msg = f"NOT_AVAILABLE: LM Studio local server not reachable at {cfg.base_url}"
        if a.json:
            _emit_json({"ok": False, "api": flavor, "error": msg})
        else:
            sys.stderr.write(msg + "\n")
        return EXIT_FAIL
    eng = _engine_if_present(paths)
    models = []
    for r in rows:
        caps = LMStudioBridge.capabilities_of(r)
        prof = eng.profiles.get(r["id"]) if eng else None
        models.append({"id": r["id"], **caps, "native_effort": prof.native_state if prof else "NONE", "stored_runs": prof.n_runs if prof else 0})
    if a.json:
        _emit_json({"ok": True, "api": flavor, "base_url": cfg.base_url, "load_state_verifiable": flavor == "native_v0", "models": models})
        return EXIT_OK
    _say(f"{len(models)} model(s) via {flavor} at {cfg.base_url}")
    for m in models:
        _say(f"  {str(m['state'] or 'unknown'):10s} {m['id']}  type={m['type']} ctx={m['max_context']} native_effort={m['native_effort']} runs={m['stored_runs']}")
    if flavor != "native_v0":
        _say("  WARNING: /api/v0/models unavailable; loaded state cannot be verified, so inference runs are refused (fail closed)")
    return EXIT_OK


def cmd_policy(a: argparse.Namespace) -> int:
    paths = _paths(a)
    eng = _engine_if_present(paths)
    cfg = eng.cfg if eng else Config.load(paths)
    out: Dict[str, Any] = {"settings": dataclasses.asdict(cfg), "hard_limits": {k: v for k, v in vars(Hard).items() if k.isupper()},
                           "efforts": EFFORT_NAMES, "database_initialized": eng is not None}
    if a.model:
        SecurityGuard.ident(a.model, "model")
        if eng is None:
            raise PlusError("not_found", "no database yet; run: python nythosplus.py install")
        out["curve"] = MCPServer(eng)._t_policy({"model_id": a.model, "family": a.family})
    if a.json:
        _emit_json(out)
        return EXIT_OK
    b = cfg.budget
    _say(f"Effort policy (minimum sufficient effort, E0..E{Hard.MAX_EFFORT})")
    _say(f"  budget       : max_effort=E{b.max_effort} calls={b.allowed_calls} tokens={b.max_tokens} elapsed={b.max_elapsed:g}s escalations={b.max_escalations}")
    _say(f"  thresholds   : enter={cfg.enter_thr} exit={cfg.exit_thr} stop_conf={cfg.stop_conf} mv_min={cfg.mv_min} min_evidence={cfg.min_evidence}")
    _say(f"  learning     : {'on' if cfg.learning_enabled else 'off'} (cold_k={cfg.cold_k:g}, min_samples={cfg.min_samples}, learned_min={cfg.learned_min})")
    if cfg.warning:
        _say(f"  WARNING      : {cfg.warning}")
    if "curve" in out:
        cv = out["curve"]
        _say(f"  {cv['model']} / {cv['family']}: kind={cv['kind']} phase={cv['phase']} samples={cv['samples']}")
        _say(f"    quality {cv['quality']}  cost {cv['cost']}  source={cv['source']}")
        _say(f"    {cv['note']}")
    elif eng is None:
        _say("  (no database yet: run install; use --model ID to see a learned curve)")
    return EXIT_OK


def cmd_benchmark(a: argparse.Namespace) -> int:
    paths = _paths(a)
    if not a.run:  # read-only summary of stored results; never calls a model
        eng = _engine_if_present(paths)
        rows = Benchmark(eng, None).summary(a.bench_id, a.model) if eng else []  # type: ignore[arg-type]
        if a.json:
            _emit_json({"ok": True, "initialized": eng is not None, "summary": rows})
            return EXIT_OK
        if not rows:
            _say("No stored benchmark results. Run one with: python nythosplus.py benchmark --run --model MODEL_ID")
            return EXIT_OK
        for x in rows:
            _say(f"  {x['mode']:9s} runs={x['runs']} scored={x['scored_runs']} mean_q={_fmt(x['mean_quality'])} first_pass={_fmt(x['first_pass_success'])} "
                 f"avg_E={_fmt(x['avg_final_effort'], 2)} q/1k_tok={_fmt(x['quality_per_1k_tokens'])}")
        return EXIT_OK
    if not a.model:
        raise PlusError("invalid_argument", "--run requires --model MODEL_ID (the model must already be loaded in LM Studio)")
    SecurityGuard.ident(a.model, "model")
    modes = [m.strip().upper() for m in a.modes.split(",") if m.strip()]
    if not modes or any(m not in RUN_MODES for m in modes):
        raise PlusError("invalid_argument", f"--modes must be a comma list from {', '.join(RUN_MODES)}")
    if not 1 <= a.repeat <= 5:
        raise PlusError("out_of_range", "--repeat must be 1..5")
    tasks = builtin_tasks()
    if a.tasks:
        want = {t.strip() for t in a.tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t["id"] in want]
        if not tasks:
            raise PlusError("invalid_argument", "no built-in task matches --tasks")
    eng = Engine(paths)
    backend = LMStudioBridge(eng.cfg.base_url, eng.cfg.http_timeout)
    progress = lambda s: sys.stderr.write(s + "\n")
    res = Benchmark(eng, PulseRunner(eng, backend)).run(a.model, modes, tasks, a.repeat, None, progress)  # explicit user action: calls the loaded model
    summ = Benchmark(eng, PulseRunner(eng, backend)).summary(res["bench_id"], a.model)
    if a.json:
        _emit_json({"ok": True, **res, "summary": summ})
        return EXIT_OK
    _say(f"benchmark {res['bench_id']}: {res['runs']} run(s)")
    for x in summ:
        _say(f"  {x['mode']:9s} runs={x['runs']} mean_q={_fmt(x['mean_quality'])} avg_E={_fmt(x['avg_final_effort'], 2)} q/1k_tok={_fmt(x['quality_per_1k_tokens'])}")
    return EXIT_OK


def cmd_history(a: argparse.Namespace) -> int:
    if not 1 <= a.limit <= 200:
        raise PlusError("out_of_range", "--limit must be 1..200")
    paths = _paths(a)
    eng = _engine_if_present(paths)
    rows = eng.history.recent(a.limit, a.bench_id) if eng else []
    keep = ("trace_id", "run_mode", "model", "family", "effort", "final_effort", "quality", "success", "escalations", "resource_state")
    runs = [dict({k: r[k] for k in keep}, time=iso(r["created_at"])) for r in rows]  # metadata only: no prompts or outputs exist in the store
    if a.json:
        _emit_json({"ok": True, "initialized": eng is not None, "runs": runs})
        return EXIT_OK
    if not runs:
        _say("No stored runs yet." if eng else "No database yet; run: python nythosplus.py install")
        return EXIT_OK
    for r in runs:
        _say(f"  {r['time']} {str(r['run_mode']):9s} {str(r['model'])[:32]:32s} {r['family']:12s} E{r['effort']}->E{r['final_effort']} q={_fmt(r['quality'])} esc={r['escalations']}")
    return EXIT_OK


_ADVISE_SCHEMA = _S({"prompt": {"type": "string", "minLength": 1, "maxLength": Hard.MAX_MCP_PROMPT},
                     "model": {"type": "string", "pattern": ID_PATTERN, "maxLength": 128},
                     "model_id": {"type": "string", "pattern": ID_PATTERN, "maxLength": 128},
                     "session": {"type": "string", "pattern": ID_PATTERN, "maxLength": 64},
                     "objective": {"type": "boolean"}}, ("prompt",))


def cmd_advise(a: argparse.Namespace) -> int:
    """Reads one bounded JSON object from stdin, prints one JSON object to stdout. Never calls a model."""
    def fail(code: str, msg: str, exit_code: int) -> int:
        if a.json:
            _emit_json({"ok": False, "error": {"code": code, "message": msg}}, compact=True)
        else:
            sys.stderr.write(f"{APP}: advise failed [{code}]: {msg}\n")
        return exit_code
    if not a.stdin:
        return fail("invalid_argument", "advise requires --stdin", EXIT_USAGE)
    try:
        raw = sys.stdin.buffer.read(ADVISE_MAX_STDIN + 1)
        if len(raw) > ADVISE_MAX_STDIN:
            raise PlusError("too_large", f"stdin exceeds {ADVISE_MAX_STDIN} bytes")
        obj = _strict_json(raw)
        if not isinstance(obj, dict):
            raise PlusError("invalid_argument", "stdin must be a JSON object")
        v = SecurityGuard.validate(_ADVISE_SCHEMA, obj)
        if "model" in v and "model_id" in v and v["model"] != v["model_id"]:
            raise PlusError("invalid_argument", "model and model_id disagree")
        model = v.get("model") or v.get("model_id") or "unknown"
        with contextlib.redirect_stdout(sys.stderr):  # stray prints can never corrupt the JSON on stdout
            eng = Engine(_paths(a))
            res = eng.advise(v["prompt"], model, v.get("session", "default"), bool(v.get("objective", False)))
    except PlusError as e:
        return fail(e.code, e.message, EXIT_USAGE if e.code in USAGE_CODES else EXIT_FAIL)
    except (sqlite3.Error, OSError):
        return fail("db_error", "state directory or database unavailable", EXIT_FAIL)
    except Exception:
        LOG.error("advise internal error: %s", traceback.format_exc(limit=3))
        return fail("internal", "internal error", EXIT_INTERNAL)
    out = {"ok": True, **res}
    if a.json:
        _emit_json(out, compact=True)
    else:
        d = res["decision"]
        _say(f"{d['effort']} ({d['mode']}, confidence {d['effort_confidence']}) family={d['task_family']} - advice only, no model call made")
        if res.get("hint"):
            _say(f"hint: {res['hint']}")
    return EXIT_OK


def run_mcp(a: argparse.Namespace) -> int:
    stdin_b, stdout_b = sys.stdin.buffer, sys.stdout.buffer  # bind the protocol streams before anything can redirect stdout
    with contextlib.redirect_stdout(sys.stderr):
        engine = Engine(_paths(a))
        if engine.cfg.warning:
            LOG.warning("settings: %s", engine.cfg.warning)
    return MCPServer(engine).serve(stdin_b, stdout_b)


# =====================================================================================
# SELF-TEST (offline, isolated temp directories, never touches real LM Studio or your data)
# =====================================================================================
class _FakeStd:
    def __init__(self, data: bytes = b""):
        self.buffer = io.BytesIO(data)

    def write(self, s: Any) -> int:
        return 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _run_cli(argv: List[str], stdin: Optional[bytes] = None) -> Tuple[int, str, str]:
    out, err, old_in = io.StringIO(), io.StringIO(), sys.stdin
    if stdin is not None:
        sys.stdin = _FakeStd(stdin)  # type: ignore[assignment]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
    finally:
        sys.stdin = old_in
    return code, out.getvalue(), err.getvalue()


def _chk(cond: Any, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class SelfTest:
    def __init__(self) -> None:
        self.rep = Report("self-test")
        self.root = Path(tempfile.mkdtemp(prefix="nythosplus-selftest-"))
        self.script = str(Paths.script())

    # ---- helpers
    def fresh(self, mcp: Optional[Dict[str, Any]] = None) -> Tuple[Path, Path]:
        d = Path(tempfile.mkdtemp(prefix="t", dir=str(self.root)))
        (d / "lm").mkdir()
        cfg = d / "lm" / "mcp.json"
        if mcp is not None:
            cfg.write_text(json.dumps(mcp, indent=2) + "\n", encoding="utf-8")
        return d / "home", cfg

    def cli(self, home: Path, cfg: Path, *argv: str, stdin: Optional[bytes] = None) -> Tuple[int, str, str]:
        return _run_cli(list(argv) + ["--home", str(home), "--config", str(cfg)], stdin)

    def env(self, home: Path) -> Dict[str, str]:
        return dict(os.environ, NYTHOSPLUS_HOME=str(home), PYTHONIOENCODING="utf-8")

    def proc(self, home: Path, args: List[str], data: Optional[bytes] = None) -> "subprocess.CompletedProcess[bytes]":
        return subprocess.run([sys.executable, self.script] + args, input=data, capture_output=True, timeout=90, env=self.env(home))

    @staticmethod
    def unrelated() -> Dict[str, Any]:
        return {"alpha": {"command": "node", "args": ["a.js"], "env": {"K": "V"}}, "beta": {"url": "http://127.0.0.1:9/mcp"}}

    def backups(self, home: Path) -> List[Path]:
        return sorted((home / "backups").glob("mcp.json.*.bak")) if (home / "backups").is_dir() else []

    # ---- tests
    def t_cli_dispatch(self) -> str:
        home, cfg = self.fresh()
        _chk(set(_DISPATCH) == set(CLI_COMMANDS), "dispatch table does not cover every command")
        code, out, _ = self.cli(home, cfg, "history", "--json")
        _chk(code == 0 and json.loads(out)["runs"] == [], "history dispatch failed")
        code, out, _ = self.cli(home, cfg, "policy", "--json")
        _chk(code == 0 and "settings" in json.loads(out), "policy dispatch failed")
        code, out, _ = self.cli(home, cfg, "benchmark", "--json")
        _chk(code == 0 and json.loads(out)["summary"] == [], "benchmark summary dispatch failed")
        code, out, _ = self.cli(home, cfg, "status", "--json", "--no-lm-studio")
        _chk(code == 0 and json.loads(out)["lm_studio"] == {"checked": False}, "status dispatch failed")
        for argv in (("history",), ("policy",), ("benchmark",), ("status", "--no-lm-studio")):  # human-readable (non-JSON) paths
            code, out, err = self.cli(home, cfg, *argv)
            _chk(code == 0 and out.strip() and "internal error" not in err, f"'{argv[0]}' without --json failed (rc={code})")
        _chk(self.cli(home, cfg, "models", "--probe", "m1", "--values", "bad")[0] == EXIT_USAGE, "models --probe must validate input before any network use")
        _chk(self.cli(home, cfg, "benchmark", "--run")[0] == EXIT_USAGE, "benchmark --run without --model must exit 2")
        _chk(_run_cli(["bogus-command"])[0] == EXIT_USAGE, "unknown command must exit 2")
        _chk(_run_cli(["--mcp", "status"])[0] == EXIT_USAGE, "--mcp combined with a command must exit 2")
        _chk(self.cli(home, cfg, "uninstall", "--purge-data")[0] == EXIT_USAGE, "--purge-data without --yes must exit 2")
        _chk(not home.exists(), "read-only commands must not create the data directory")
        p = self.proc(home, [])
        _chk(p.returncode == 0 and b"usage" in p.stdout.lower(), "no-argument run must print usage and exit 0")
        _chk(self.proc(home, ["bogus"]).returncode == EXIT_USAGE, "subprocess: unknown command must exit 2")
        return "in-process and subprocess dispatch, exit codes 0/2 OK"

    def t_help(self) -> str:
        code, out, _ = _run_cli(["--help"])
        _chk(code == 0, "--help must exit 0")
        for c in CLI_COMMANDS:
            _chk(c in out, f"--help does not list '{c}'")
        _chk("--mcp" in out, "--help does not list --mcp")
        code, out, _ = _run_cli(["install", "--help"])
        _chk(code == 0 and "--dry-run" in out, "install --help incomplete")
        p = self.proc(self.root / "h", ["--help"])
        _chk(p.returncode == 0 and b"install" in p.stdout, "subprocess --help failed")
        return "top-level and per-command help OK"

    def t_database_init(self) -> str:
        home, cfg = self.fresh({"mcpServers": {}})
        _chk(self.cli(home, cfg, "install", "--dry-run")[0] == 0, "dry-run failed")
        _chk(not home.exists(), "dry-run created files")
        code, out, _ = self.cli(home, cfg, "install")
        _chk(code == 0, f"install failed: {out[-300:]}")
        st = Store(home / "nythosplus.db")
        _chk(st.schema_version() == SCHEMA_VERSION, "schema version mismatch")
        _chk(st.integrity()[0], "integrity check failed")
        _chk((home / "config.json").exists(), "state file not created")
        with st.read() as c:
            cols = [r[1] for t in [x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")] for r in c.execute(f"PRAGMA table_info({t})")]
        _chk(not any(re.search(r"chain|thought|thinking|activation|hidden|scratch|prompt|output_text|answer_text|completion", x) for x in cols),
             "schema has a column that could hold prompts, hidden activations or chain-of-thought")
        return f"database initialized (schema v{SCHEMA_VERSION}, integrity ok, no prompt/CoT/activation columns)"

    def t_install_dry_run(self) -> str:
        home, cfg = self.fresh({"mcpServers": self.unrelated()})
        before = cfg.read_bytes()
        code, out, _ = self.cli(home, cfg, "install", "--dry-run")
        _chk(code == 0 and "would" in out, "dry-run did not report planned steps")
        _chk(cfg.read_bytes() == before, "dry-run modified mcp.json")
        _chk(not home.exists(), "dry-run wrote Nythos Plus files")
        return "dry-run reports steps and writes nothing"

    def t_plugin_generation(self) -> str:
        home, cfg = self.fresh({"mcpServers": {}})
        _chk(self.cli(home, cfg, "install")[0] == 0, "install failed")
        pd = home / "plugin" / "nythosplus"
        marker = json.loads((pd / PLUGIN_MARKER).read_text("utf-8"))
        _chk(marker["owner"].startswith(OWNER_PREFIX), "ownership marker missing")
        for n, h in marker["files"].items():
            _chk(sha256_bytes((pd / n).read_bytes()) == h, f"{n} hash mismatch")
        on_disk = {p.relative_to(pd).as_posix() for p in pd.rglob("*") if p.is_file()}
        _chk(on_disk == set(marker["files"]) | {PLUGIN_MARKER}, "plugin folder holds files not listed in the marker")
        _chk(PluginBuilder.validate(PluginBuilder(sys.executable, self.script, "0" * 32).files()) == [], "generated plugin failed validation")
        ts = (pd / "src" / "advise.ts").read_text("utf-8")
        _chk("shell: false" in ts and '"--stdin"' in ts, "plugin does not call advise safely")
        # a foreign folder must never be written into
        home2, cfg2 = self.fresh({"mcpServers": {}})
        fp = home2 / "plugin" / "nythosplus"
        fp.mkdir(parents=True)
        (fp / "mine.txt").write_text("user file", encoding="utf-8")
        code, _, _ = self.cli(home2, cfg2, "install")
        _chk(code == EXIT_FAIL, "install must fail when the plugin folder is not Nythos Plus-owned")
        _chk([p.name for p in fp.iterdir()] == ["mine.txt"], "install wrote into an unowned plugin folder")
        return "plugin generated, hashed, validated; unowned folder refused"

    def t_install_backup(self) -> str:
        orig = {"mcpServers": self.unrelated(), "extra": {"keep": True}}
        home, cfg = self.fresh(orig)
        raw = cfg.read_bytes()
        code, out, _ = self.cli(home, cfg, "install")
        _chk(code == 0, f"install failed: {out[-300:]}")
        bk = self.backups(home)
        _chk(len(bk) == 1 and bk[0].read_bytes() == raw, "backup missing or not identical to the original")
        d = json.loads(cfg.read_text("utf-8"))
        _chk(d["extra"] == orig["extra"] and all(d["mcpServers"][k] == v for k, v in orig["mcpServers"].items()), "unrelated configuration changed")
        e = d["mcpServers"][SERVER_KEY]
        _chk(Installer.is_owned(e) and e["args"][-1] == "--mcp", "entry not owned/valid")
        _chk(len([k for k, v in d["mcpServers"].items() if Installer.is_owned(v)]) == 1, "entry not registered exactly once")
        after = cfg.read_bytes()
        _chk(self.cli(home, cfg, "install")[0] == 0 and cfg.read_bytes() == after and len(self.backups(home)) == 1, "second install must be a no-op without a new backup")
        home2, cfg2 = self.fresh()
        _chk(self.cli(home2, cfg2, "install")[0] == 0 and cfg2.exists() and not self.backups(home2), "fresh install should create mcp.json without a backup")
        return "backup is byte-identical, unrelated entries preserved, install idempotent"

    def t_ownership_protection(self) -> str:
        foreign = {"mcpServers": {"other": {"command": "x", "args": []}, SERVER_KEY: {"command": "node", "args": ["mine.js"]}}}
        home, cfg = self.fresh(foreign)
        raw = cfg.read_bytes()
        _chk(self.cli(home, cfg, "install")[0] == EXIT_FAIL, "install must refuse an unowned entry")
        _chk(cfg.read_bytes() == raw and not self.backups(home), "install touched an unowned entry")
        _chk(self.cli(home, cfg, "uninstall")[0] == EXIT_FAIL, "uninstall must refuse an unowned entry")
        _chk(cfg.read_bytes() == raw, "uninstall touched an unowned entry")
        self.cli(home, cfg, "repair")
        _chk(cfg.read_bytes() == raw, "repair touched an unowned entry")
        cfg.write_text("{ not json", encoding="utf-8")
        _chk(self.cli(home, cfg, "install")[0] == EXIT_FAIL and cfg.read_text() == "{ not json", "invalid mcp.json must fail and stay untouched")
        return "unowned 'nythosplus' entry and invalid JSON are never modified (install/uninstall/repair)"

    def t_rollback(self) -> str:
        class Flaky(Installer):
            orig: Optional[bytes] = None

            def _load(self, path: Path) -> Tuple[Dict[str, Any], Optional[bytes]]:
                d, raw = super()._load(path)
                if self.orig is not None and raw is not None and raw != self.orig:
                    return {"mcpServers": {}}, raw  # simulate a post-write verification mismatch
                return d, raw
        home, cfg = self.fresh({"mcpServers": self.unrelated()})
        raw = cfg.read_bytes()
        inst = Flaky(Paths(home), str(cfg))
        inst.orig = raw
        rep = inst.install()
        _chk(rep.failed and any(n == "rollback" for _, n, _ in rep.rows), "failed verification must report a rollback")
        _chk(cfg.read_bytes() == raw, "install rollback did not restore the original mcp.json")
        _chk(len(self.backups(home)) == 1, "backup missing after rollback")
        good_home, good_cfg = self.fresh({"mcpServers": self.unrelated()})
        Installer(Paths(good_home), str(good_cfg)).install()
        installed = good_cfg.read_bytes()
        inst2 = Flaky(Paths(good_home), str(good_cfg))
        inst2.orig = installed
        rep2 = inst2.uninstall()
        _chk(rep2.failed and good_cfg.read_bytes() == installed, "uninstall rollback did not restore the original mcp.json")
        return "failed verification restores the original mcp.json (install and uninstall)"

    def t_uninstall_preservation(self) -> str:
        orig = {"mcpServers": self.unrelated(), "extra": 1}
        home, cfg = self.fresh(orig)
        _chk(self.cli(home, cfg, "install")[0] == 0, "install failed")
        pd = home / "plugin" / "nythosplus"
        (pd / "user-notes.txt").write_text("keep me", encoding="utf-8")
        snap = cfg.read_bytes()
        _chk(self.cli(home, cfg, "uninstall", "--dry-run")[0] == 0 and cfg.read_bytes() == snap and (pd / PLUGIN_MARKER).exists(), "uninstall dry-run changed something")
        code, out, _ = self.cli(home, cfg, "uninstall")
        _chk(code == 0, f"uninstall failed: {out[-300:]}")
        d = json.loads(cfg.read_text("utf-8"))
        _chk(SERVER_KEY not in d["mcpServers"] and d["mcpServers"] == orig["mcpServers"] and d["extra"] == 1, "uninstall changed unrelated configuration")
        _chk(not (pd / PLUGIN_MARKER).exists() and not (pd / "src" / "index.ts").exists(), "owned plugin files remain")
        _chk((pd / "user-notes.txt").read_text() == "keep me", "uninstall removed a file it does not own")
        _chk((home / "nythosplus.db").exists() and (home / "config.json").exists(), "uninstall deleted user data")
        _chk(len(self.backups(home)) == 2, "uninstall did not create a backup")
        home2, cfg2 = self.fresh({"mcpServers": {}})
        self.cli(home2, cfg2, "install")
        code, _, _ = self.cli(home2, cfg2, "uninstall", "--purge-data", "--yes")
        _chk(code == 0 and not (home2 / "nythosplus.db").exists() and (home2 / "backups").is_dir(), "explicit --purge-data --yes should delete data but keep backups")
        return "only owned files/entry removed; user data and unrelated servers preserved; purge needs --yes"

    def t_repair(self) -> str:
        home, cfg = self.fresh({"mcpServers": self.unrelated()})
        _chk(self.cli(home, cfg, "install")[0] == 0, "install failed")
        d = json.loads(cfg.read_text("utf-8"))
        d["mcpServers"][SERVER_KEY]["command"] = "old-python"
        d["mcpServers"]["zeta"] = {"command": "z", "args": []}
        cfg.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
        adv = home / "plugin" / "nythosplus" / "src" / "advise.ts"
        adv.write_text(adv.read_text("utf-8") + "\n// tampered\n", encoding="utf-8")
        stale = home / (".old." + TMP_SUFFIX)
        stale.write_text("x", encoding="utf-8")
        os.utime(stale, (now() - 5000, now() - 5000))
        before = len(self.backups(home))
        code, out, _ = self.cli(home, cfg, "repair")
        _chk(code == 0, f"repair failed: {out[-300:]}")
        _chk(not stale.exists(), "stale temp file not removed")
        _chk(Installer(Paths(home), str(cfg)).plugin_status()["intact"], "plugin not regenerated")
        d2 = json.loads(cfg.read_text("utf-8"))
        _chk(d2["mcpServers"][SERVER_KEY]["command"] == sys.executable, "registration not repaired")
        _chk(all(d2["mcpServers"][k] == d["mcpServers"][k] for k in ("alpha", "beta", "zeta")), "repair changed unrelated servers")
        _chk(len(self.backups(home)) == before + 1, "repair did not back up mcp.json before modifying it")
        shutil.rmtree(home / "plugin")
        _chk(self.cli(home, cfg, "repair")[0] == 0 and (home / "plugin" / "nythosplus" / PLUGIN_MARKER).exists(), "missing plugin not regenerated")
        return "db checked, temp files cleaned, plugin and registration repaired, unrelated servers untouched"

    def t_advise_stdin(self) -> str:
        home, cfg = self.fresh()
        good = json.dumps({"prompt": "Compute 17*23 + 144/12. Answer with the number only.", "model": "unknown"}).encode()
        orig = (LMStudioBridge._get, LMStudioBridge.chat, socket.socket.connect)

        def trip(*_a: Any, **_k: Any) -> Any:
            raise AssertionError("advise attempted model or network access")
        LMStudioBridge._get, LMStudioBridge.chat, socket.socket.connect = trip, trip, trip  # type: ignore[assignment,method-assign]
        try:
            code, out, _ = self.cli(home, cfg, "advise", "--stdin", "--json", stdin=good)
        finally:
            LMStudioBridge._get, LMStudioBridge.chat, socket.socket.connect = orig  # type: ignore[method-assign]
        _chk(code == 0, f"in-process advise failed: {out[-200:]}")
        j = json.loads(out)
        _chk(j["ok"] is True and j["model_call_made"] is False and "decision" in j, "advise output malformed")
        p = self.proc(home, ["advise", "--stdin", "--json"], good)
        _chk(p.returncode == 0, "subprocess advise failed")
        j = json.loads(p.stdout.decode("utf-8"))  # stdout must be exactly one JSON document
        _chk(j["ok"] is True and j["model_call_made"] is False, "subprocess advise output malformed")
        cases = [(b"{not json", "invalid_json"), (b"", "invalid_json"), (b"[1]", "invalid_argument"),
                 (json.dumps({"prompt": "x", "bogus": 1}).encode(), "unknown_argument"), (json.dumps({"prompt": ""}).encode(), "invalid_argument"),
                 (json.dumps({"prompt": "a" * 9000}).encode(), "too_long"), (json.dumps({"prompt": "a\u0001b"}).encode(), "invalid_argument"),
                 (json.dumps({"prompt": "x", "model": "../etc"}).encode(), "invalid_id"), (b"x" * (ADVISE_MAX_STDIN + 10), "too_large")]
        for data, code_expected in cases:
            p = self.proc(home, ["advise", "--stdin", "--json"], data)
            r = json.loads(p.stdout.decode("utf-8"))
            _chk(p.returncode != 0 and r["ok"] is False and r["error"]["code"] == code_expected, f"bad input not rejected as {code_expected}")
        return "bounded stdin, machine-readable JSON only, never contacts a model or the network, bad input exits non-zero"

    def t_mcp_startup(self) -> str:
        ok, msg = mcp_probe()
        _chk(ok, f"subprocess probe failed: {msg}")
        home, cfg = self.fresh()
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"}, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "nythosplus_status", "arguments": {}}}]
        fin, fout = _FakeStd(b"".join(json.dumps(m).encode() + b"\n" for m in msgs)), _FakeStd()
        old_in, old_out = sys.stdin, sys.stdout
        sys.stdin, sys.stdout = fin, fout  # type: ignore[assignment]
        try:
            code = main(["--mcp", "--home", str(home)])
        finally:
            sys.stdin, sys.stdout = old_in, old_out
        lines = [json.loads(x) for x in fout.buffer.getvalue().split(b"\n") if x.strip()]
        _chk(code == 0 and [m["id"] for m in lines] == [1, 2, 3], "--mcp did not run the existing MCPServer protocol loop")
        _chk(len(lines[1]["result"]["tools"]) == len(MCPServer(Engine(Paths(home))).tools), "tool list mismatch")
        return f"--mcp starts MCPServer; stdout protocol-only ({msg})"

    def t_security(self) -> str:
        for bad in ("http://example.com:80", "http://192.168.0.5:1234", "https://127.0.0.1:1234", "http://127.0.0.1:1234/x", "http://u:p@127.0.0.1:1234"):
            try:
                SecurityGuard.local_url(bad)
            except PlusError:
                continue
            raise AssertionError(f"non-loopback URL accepted: {bad}")
        home, _cfg = self.fresh()
        home.mkdir(parents=True)
        (home / "config.json").write_text(json.dumps({"settings": {"base_url": "http://example.com:1234"}}), encoding="utf-8")
        _chk(Config.load(Paths(home)).base_url == "http://127.0.0.1:1234", "remote base_url in config was accepted")
        tree = ast.parse(Path(self.script).read_text("utf-8"))
        patt = re.compile("/(?:un" + "load|load|down" + "load)\\b|\\blms\\s+(?:un" + "load|load|get)\\b")
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                for kw in n.keywords:
                    _chk(kw.arg != "shell" or (isinstance(kw.value, ast.Constant) and kw.value.value is False), "shell=True found")
                f = n.func
                _chk(not (isinstance(f, ast.Name) and f.id in ("eval", "exec")), "eval/exec found")
                _chk(not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "os" and f.attr in ("system", "popen")), "os.system/popen found")
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                _chk(not patt.search(n.value), "a string constant references model-lifecycle endpoints")
        for name in ("load", "unload", "load_model", "unload_model", "switch_model", "download", "download_model"):
            _chk(not hasattr(ModelBackend, name) and not hasattr(LMStudioBridge, name), f"model-control method '{name}' exists")

        class V1Only(LMStudioBridge):
            def list_models(self) -> Tuple[str, List[Dict[str, Any]]]:
                return "openai_v1", [{"id": "m", "state": "unknown"}]

        class Down(LMStudioBridge):
            def list_models(self) -> Tuple[str, List[Dict[str, Any]]]:
                return "unavailable", []

        class Idle(LMStudioBridge):
            def list_models(self) -> Tuple[str, List[Dict[str, Any]]]:
                return "native_v0", [{"id": "m", "state": "not-loaded"}]

        class Loaded(LMStudioBridge):
            def list_models(self) -> Tuple[str, List[Dict[str, Any]]]:
                return "native_v0", [{"id": "m", "state": "loaded"}]
        for cls, want in ((V1Only, "model_state_unknown"), (Down, "model_state_unknown"), (Idle, "model_not_loaded")):
            try:
                cls("http://127.0.0.1:1234").ensure_loaded("m")
            except PlusError as e:
                _chk(e.code == want, f"{cls.__name__}: expected {want}, got {e.code}")
                continue
            raise AssertionError(f"{cls.__name__}: unverifiable/unloaded model was accepted (must fail closed)")
        Loaded("http://127.0.0.1:1234").ensure_loaded("m")
        try:
            PulseRunner(Engine(Paths(home)), V1Only("http://127.0.0.1:1234")).run("hello", "m")
            raise AssertionError("runner proceeded without a verified loaded model")
        except PlusError as e:
            _chk(e.code in ("model_state_unknown", "resource_critical"), f"runner failed with unexpected code {e.code}")
        eng = Engine(Paths(home))
        tid = eng.advise("PROMPTCANARY91 compute 2+2", "m")["decision"]["trace_id"]
        MCPServer(eng)._t_evaluate({"trace_id": tid, "output": "OUTPUTCANARY47", "checks": ["contains:OUTPUTCANARY"]})
        blob = b"".join(p.read_bytes() for p in home.glob("nythosplus.db*") if p.is_file())
        _chk(b"OUTPUTCANARY47" not in blob, "evaluated output text was stored")
        _chk(b"PROMPTCANARY91" not in blob, "prompt text was stored")
        return "loopback-only, shell=False, no eval/exec, no model-control API, fail-closed load check, no output/prompt storage"

    TESTS = ("cli_dispatch", "help", "database_init", "install_dry_run", "plugin_generation", "install_backup", "ownership_protection",
             "rollback", "uninstall_preservation", "repair", "advise_stdin", "mcp_startup", "security")

    def run(self) -> Report:
        try:
            for name in self.TESTS:
                try:
                    self.rep.ok(name, getattr(self, "t_" + name)() or "ok")
                except AssertionError as e:
                    self.rep.fail(name, str(e) or "assertion failed")
                except Exception as e:  # a crashing test is a failing test, never a crash of the runner
                    self.rep.fail(name, f"{type(e).__name__}: {str(e)[:160]}")
            self.rep.skip("lm_studio_live", "NOT VERIFIED LIVE: self-test is offline and never launches the plugin or MCP server from LM Studio")
        finally:
            shutil.rmtree(self.root, ignore_errors=True)
        return self.rep


# =====================================================================================
# ARGUMENT PARSER / MAIN
# =====================================================================================
_DISPATCH: Dict[str, Callable[[argparse.Namespace], int]] = {
    "status": cmd_status, "doctor": cmd_doctor, "self-test": cmd_self_test, "install": cmd_install, "repair": cmd_repair,
    "uninstall": cmd_uninstall, "models": cmd_models, "policy": cmd_policy, "benchmark": cmd_benchmark, "history": cmd_history, "advise": cmd_advise,
}


def _common(p: argparse.ArgumentParser) -> None:
    S = argparse.SUPPRESS
    p.add_argument("--home", default=S, metavar="DIR", help="Nythos Plus data directory (default: NYTHOSPLUS_HOME or the platform default)")
    p.add_argument("--config", default=S, metavar="MCP_JSON", help="path to LM Studio mcp.json (file must be named mcp.json)")
    p.add_argument("--json", action="store_true", default=S, help="machine-readable JSON on stdout")
    p.add_argument("-v", "--verbose", action="store_true", default=S, help="log INFO messages to stderr")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nythosplus.py", description=f"{APP} {VERSION} - {TAGLINE}. Advice and evaluation only: it never loads, unloads or switches models.",
                                epilog="Live LM Studio integration is NOT VERIFIED by this tool unless you test it yourself (see: lms dev).")
    p.add_argument("--mcp", action="store_true", help="run the MCP server over stdio (stdout is protocol-only; logs go to stderr)")
    p.add_argument("--version", action="version", version=f"{APP} {VERSION}")
    _common(p)
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, description=help_)
        _common(sp)
        return sp
    s = add("status", "show database, registration, plugin and LM Studio status")
    s.add_argument("--no-lm-studio", action="store_true", help="do not contact the LM Studio local server")
    add("doctor", "run read-only diagnostics (includes a subprocess MCP protocol probe; no inference)")
    add("self-test", "run the offline self-test suite in isolated temp directories")
    i = add("install", "initialize state, generate the LM Studio plugin and register the MCP server")
    i.add_argument("--dry-run", action="store_true", help="report what would happen; write nothing")
    i.add_argument("--skip-plugin", action="store_true", help="do not generate the plugin files")
    i.add_argument("--skip-mcp", action="store_true", help="do not touch mcp.json")
    add("repair", "check the database, clean temp files, regenerate owned plugin files, repair the owned registration")
    u = add("uninstall", "remove only Nythos Plus-owned plugin files and the owned mcp.json entry (data is kept)")
    u.add_argument("--dry-run", action="store_true", help="report what would be removed; change nothing")
    u.add_argument("--purge-data", action="store_true", help="DESTRUCTIVE: also delete the database and config.json (requires --yes; backups are kept)")
    u.add_argument("--yes", action="store_true", help="confirm destructive options")
    m = add("models", "list LM Studio models and their load state (read-only unless --probe)")
    m.add_argument("--probe", metavar="MODEL_ID", help="explicit opt-in: two tiny inference calls to an already-loaded model to test native effort control")
    m.add_argument("--values", metavar="LOW,MID,HIGH", help="native effort values to probe (default low,medium,high)")
    po = add("policy", "show effective effort policy settings and, with --model, the learned curve")
    po.add_argument("--model", metavar="MODEL_ID")
    po.add_argument("--family", default="general", choices=list(FAMILIES))
    b = add("benchmark", "summarize stored benchmark results; with --run, benchmark a loaded model")
    b.add_argument("--run", action="store_true", help="explicit opt-in: run the built-in tasks against a loaded model")
    b.add_argument("--model", metavar="MODEL_ID")
    b.add_argument("--modes", default="RAW,PULSE", help=f"comma list from {','.join(RUN_MODES)} (default RAW,PULSE)")
    b.add_argument("--repeat", type=int, default=1, help="repetitions 1..5")
    b.add_argument("--tasks", metavar="IDS", help="comma list of built-in task ids")
    b.add_argument("--bench-id", metavar="ID", help="summarize one stored benchmark")
    h = add("history", "show recent stored run metadata (no prompts or outputs are stored)")
    h.add_argument("--limit", type=int, default=10, help="1..200")
    h.add_argument("--bench-id", metavar="ID")
    ad = add("advise", "read a bounded JSON object from stdin and print effort advice; never calls a model")
    ad.add_argument("--stdin", action="store_true", help='read {"prompt": "...", "model": "..."} from stdin (max %d bytes)' % ADVISE_MAX_STDIN)
    return p


def _main(argv: Optional[List[str]]) -> int:
    parser = build_parser()
    try:
        a = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (EXIT_OK if e.code is None else EXIT_USAGE)
    for k, dflt in (("json", False), ("verbose", False), ("home", None), ("config", None)):  # SUPPRESS defaults keep sub-command flags from clobbering top-level ones
        if not hasattr(a, k):
            setattr(a, k, dflt)
    _setup_logging(bool(a.verbose))
    try:
        if a.mcp:
            if a.command:
                sys.stderr.write(f"{APP}: --mcp cannot be combined with a command\n")
                return EXIT_USAGE
            return run_mcp(a)
        if not a.command:
            _say(render_logo(Style()))
            _say(f"\n  {APP} {VERSION} - {TAGLINE}\n")
            parser.print_help()
            return EXIT_OK
        return _DISPATCH[a.command](a)
    except PlusError as e:
        sys.stderr.write(f"{APP}: error [{e.code}]: {e.message}\n")
        return EXIT_USAGE if e.code in USAGE_CODES else EXIT_FAIL
    except KeyboardInterrupt:
        sys.stderr.write(f"{APP}: interrupted\n")
        return EXIT_INTERRUPT
    except (sqlite3.Error, OSError) as e:
        sys.stderr.write(f"{APP}: error: {type(e).__name__}: {str(e)[:200]}\n")
        return EXIT_FAIL
    except Exception:
        LOG.error("internal error: %s", traceback.format_exc(limit=5))
        sys.stderr.write(f"{APP}: internal error (details above); no state was deleted\n")
        return EXIT_INTERNAL


def main(argv: Optional[List[str]] = None) -> int:
    saved = os.environ.get("NYTHOSPLUS_HOME")
    try:
        return _main(argv)
    finally:
        if saved is None:
            os.environ.pop("NYTHOSPLUS_HOME", None)
        else:
            os.environ["NYTHOSPLUS_HOME"] = saved


if __name__ == "__main__":
    sys.exit(main())
