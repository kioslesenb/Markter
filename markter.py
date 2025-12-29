import os
import time
import json
import hashlib
import sqlite3
import threading
import requests
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from gradio_client import Client

# ----------------- ENV -----------------
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "").strip()

# ----------------- settings -----------------
DB_PATH = "cache.sqlite3"
TTL_SECONDS_DEFAULT = 6 * 60 * 60
TTL_SECONDS_FALLBACK = 24 * 60 * 60
MIN_SECONDS_BETWEEN_CALLS = 2
MAX_CALLS_PER_DAY = 200

# ----------------- app -----------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


# ----------------- global limiter state -----------------
_limiter_lock = threading.Lock()
_last_call_ts = 0.0
_day_key = None
_day_calls = 0


def _today_key_local() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def _limiter_allow() -> tuple[bool, dict]:
    global _last_call_ts, _day_key, _day_calls
    with _limiter_lock:
        now = time.time()
        tk = _today_key_local()
        if _day_key != tk:
            _day_key = tk
            _day_calls = 0
            _last_call_ts = 0.0
        wait_needed = max(0.0, MIN_SECONDS_BETWEEN_CALLS - (now - _last_call_ts))
        if _day_calls >= MAX_CALLS_PER_DAY:
            return False, {"reason": "daily_cap", "day_calls": _day_calls, "day_cap": MAX_CALLS_PER_DAY}
        if wait_needed > 0:
            return False, {"reason": "min_interval", "retry_after_seconds": int(wait_needed) + 1}
        _last_call_ts = now
        _day_calls += 1
        return True, {"day_calls": _day_calls, "day_cap": MAX_CALLS_PER_DAY}


# ----------------- sqlite cache -----------------
_db_lock = threading.Lock()


def _db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            k TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            lim INTEGER NOT NULL,
            ts INTEGER NOT NULL,
            payload TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_ts ON cache(ts)")
    return conn


_conn = _db()


def _cache_key(query: str, limit: int) -> str:
    return hashlib.sha256(f"{query}\n{limit}".encode("utf-8")).hexdigest()


def _cache_get(query: str, limit: int):
    k = _cache_key(query, limit)
    with _db_lock:
        row = _conn.execute("SELECT ts, payload FROM cache WHERE k=?", (k,)).fetchone()
    if not row:
        return None
    ts, payload = row
    try:
        data = json.loads(payload)
    except Exception:
        return None
    age = int(time.time()) - int(ts)
    data.setdefault("debug", {})
    data["debug"]["cache"] = "HIT"
    data["debug"]["cache_age_seconds"] = age
    return {"k": k, "ts": int(ts), "age": age, "data": data}


def _cache_set(query: str, limit: int, data: dict):
    k = _cache_key(query, limit)
    ts = int(time.time())
    payload = json.dumps(data, ensure_ascii=False)
    with _db_lock:
        _conn.execute(
            "INSERT OR REPLACE INTO cache(k, query, lim, ts, payload) VALUES(?,?,?,?,?)",
            (k, query, int(limit), ts, payload),
        )
        _conn.commit()


def _cache_clear():
    with _db_lock:
        _conn.execute("DELETE FROM cache")
        _conn.commit()


# ----------------- HF Space client (lazy) -----------------
_hf_client = None
_hf_lock = threading.Lock()


def _get_hf_client() -> Client:
    global _hf_client
    if not HF_SPACE_URL:
        raise RuntimeError("HF_SPACE_URL is missing")
    with _hf_lock:
        if _hf_client is None:
            _hf_client = Client(HF_SPACE_URL)
        return _hf_client


def generate_keywords_from_image(image_bytes: bytes, filename: str = "image.png") -> str:
    """
    Uses Hugging Face Space (Gradio) to generate a caption/query from an image.
    Sends the image as an uploaded file tuple (filename, filehandle) to satisfy ImageData validation.
    """
    if not HF_SPACE_URL:
        return "used product"

    try:
        client = _get_hf_client()

        # keep extension if we have one
        suffix = ".png"
        if filename and "." in filename:
            suffix = "." + filename.rsplit(".", 1)[1].lower()

        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(image_bytes)
            f.flush()
            with open(f.name, "rb") as fp:
                out = client.predict((filename or f"image{suffix}", fp))

        if isinstance(out, str) and out.strip():
            return out.strip()
        return "used product"
    except Exception as e:
        print("HF Space error:", repr(e))
        return "used product"


# ----------------- serpapi helpers -----------------
def _serpapi_call_sold(query: str, limit: int):
    params = {
        "engine": "ebay",
        "ebay_domain": "ebay.nl",
        "_nkw": query,
        "filters": "Sold,Complete",
        "_ipg": "200",
        "api_key": SERPAPI_KEY,
    }
    backoffs = [2, 4, 8]
    last_debug = {}
    for attempt, wait_s in enumerate(backoffs, start=1):
        r = requests.get("https://serpapi.com/search", params=params, timeout=30)
        last_debug = {
            "attempt": attempt,
            "status_code": r.status_code,
            "ebay_url": r.json().get("search_metadata", {}).get("ebay_url", "N/A") if r.ok else "N/A",
        }
        if r.status_code >= 500:
            time.sleep(wait_s)
            continue
        break

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail={"reason": "serpapi_failed", "debug": last_debug})

    data = r.json()
    if "error" in data:
        raise HTTPException(status_code=502, detail={"reason": "serpapi_error", "error": data["error"], "debug": last_debug})

    items = data.get("organic_results", []) or []
    comps = []
    for it in items:
        price_info = it.get("price", {}) or {}
        extracted = price_info.get("extracted_value") or price_info.get("extracted")
        raw = price_info.get("raw", "")
        currency = (price_info.get("currency") or "").upper()

        if extracted is None:
            continue

        is_eur = currency == "EUR" or "€" in raw or "EUR" in raw.upper()
        if not is_eur:
            continue

        try:
            price_eur = float(extracted)
        except Exception:
            continue

        comps.append({
            "title": (it.get("title", "") or "")[:200],
            "price_eur": price_eur,
            "url": it.get("link", ""),
            "condition": it.get("condition", "Onbekend"),
            "source": "ebay_sold_serpapi",
        })

    return comps[:limit], {
        "count_raw": len(items),
        "count_eur": len(comps),
        "serpapi_debug": last_debug
    }


# ----------------- main fetch -----------------
def fetch_ebay_sold(query: str, limit: int = 50):
    if not SERPAPI_KEY:
        raise HTTPException(status_code=400, detail={"reason": "missing_serpapi_key"})

    limit = max(1, min(int(limit), 200))

    cached = _cache_get(query, limit)
    if cached and cached["age"] <= TTL_SECONDS_DEFAULT:
        cached["data"]["debug"].update({"ttl_seconds": TTL_SECONDS_DEFAULT, "stale": False})
        return cached["data"]

    allowed, lim_dbg = _limiter_allow()
    if not allowed:
        if cached:
            cached["data"]["debug"].update({"ttl_seconds": TTL_SECONDS_DEFAULT, "stale": True, "refresh_skipped": lim_dbg})
            return cached["data"]
        raise HTTPException(status_code=429, detail={"reason": "local_rate_limited", "debug": lim_dbg})

    try:
        comps, dbg = _serpapi_call_sold(query, limit)
        out = {"items": comps, "debug": {"cache": "MISS", "ttl_seconds": TTL_SECONDS_DEFAULT} | dbg}
        _cache_set(query, limit, out)
        return out
    except HTTPException as e:
        if cached:
            cached["data"]["debug"].update({"ttl_seconds": TTL_SECONDS_FALLBACK, "stale": True, "refresh_failed": e.detail})
            return cached["data"]
        raise


# ----------------- analyze endpoint -----------------
@app.post("/analyze")
async def analyze_photo(file: UploadFile = File(...)):
    image_bytes = await file.read()
    filename = file.filename or "image.png"

    query = generate_keywords_from_image(image_bytes, filename=filename)

    limit = 30
    result = fetch_ebay_sold(query, limit=limit)

    prices = [item["price_eur"] for item in result.get("items", []) if "price_eur" in item]
    if prices:
        avg = sum(prices) / len(prices)
        result["stats"] = {
            "average_eur": round(avg, 2),
            "min_eur": min(prices),
            "max_eur": max(prices),
            "count": len(prices)
        }

    result["used_query"] = query
    return result


# ----------------- debug endpoints -----------------
@app.get("/ping")
def ping():
    return {"ok": True}


@app.get("/debug/ebay")
def debug_ebay(q: str, limit: int = 50):
    return fetch_ebay_sold(q, limit=limit)


@app.get("/debug/cache_clear")
def cache_clear():
    _cache_clear()
    return {"ok": True, "message": "Cache cleared"}


print("\n=== ROUTES LOADED ===")
for r in app.routes:
    if isinstance(r, APIRoute):
        print(f"{','.join(sorted(r.methods))}\t{r.path}\t->\t{r.name}")
print("=====================\n")
