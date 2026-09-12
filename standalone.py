"""
standalone.py — Single-process, no-Docker runner for the flood response system.

Replaces PostGIS/Redis with SQLite + in-memory pub/sub so the demo runs with
a single `python standalone.py` command. The background simulators, risk,
spread, and optimizer workers run as threads inside the same process.
"""
from __future__ import annotations

import asyncio
import json
import math
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.app.core.config import settings, BASIN


# ══════════════════════════════════════════════════════════════════
# In-memory event bus (replaces Redis Streams for standalone mode)
# ══════════════════════════════════════════════════════════════════

class MemoryBus:
    def __init__(self):
        self._subs: dict[str, list[queue.Queue]] = {}
        self._latest: dict[str, dict] = {}
        self._kvs: dict[str, str] = {}
        self._lock = threading.Lock()

    def publish(self, channel: str, payload: dict):
        payload = dict(payload)
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        self._latest[channel] = payload
        with self._lock:
            subs = list(self._subs.get(channel, []))
        for q in subs:
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

    def latest(self, channel: str) -> dict | None:
        return self._latest.get(channel)

    def set_kv(self, key: str, val: str):
        self._kvs[key] = val

    def get_kv(self, key: str) -> str | None:
        return self._kvs.get(key)

    def listen(self, channel: str, timeout: float = 2.0) -> dict | None:
        # Thread-local subscription queue
        with self._lock:
            q = queue.Queue(maxsize=100)
            self._subs.setdefault(channel, []).append(q)
        try:
            try:
                return q.get(timeout=timeout)
            except queue.Empty:
                return None
        finally:
            with self._lock:
                self._subs[channel] = [s for s in self._subs[channel] if s is not q]


bus = MemoryBus()


# ══════════════════════════════════════════════════════════════════
# SQLite-backed state store (replaces PostGIS)
# ══════════════════════════════════════════════════════════════════

STATE_PATH = settings.data_dir / "standalone_state.db"


def get_conn():
    conn = sqlite3.connect(str(STATE_PATH), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS settlements (
        id INTEGER PRIMARY KEY, osm_id INTEGER UNIQUE, name TEXT, population INTEGER,
        lon REAL, lat REAL);
    CREATE TABLE IF NOT EXISTS rivers (id INTEGER PRIMARY KEY, osm_id INTEGER, name TEXT,
        waterway TEXT, geojson TEXT);
    CREATE TABLE IF NOT EXISTS roads (id INTEGER PRIMARY KEY, osm_id INTEGER, name TEXT,
        highway TEXT, geojson TEXT);
    CREATE TABLE IF NOT EXISTS pois (id INTEGER PRIMARY KEY, osm_id INTEGER, name TEXT,
        amenity TEXT, category TEXT, lon REAL, lat REAL);
    CREATE TABLE IF NOT EXISTS gauges (id INTEGER PRIMARY KEY, code TEXT UNIQUE, name TEXT,
        type TEXT, elevation REAL, lon REAL, lat REAL);
    CREATE TABLE IF NOT EXISTS rainfall_obs (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, gauge_id INTEGER, value_mmhr REAL, quality TEXT DEFAULT 'good');
    CREATE TABLE IF NOT EXISTS water_level_obs (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, gauge_id INTEGER, level_m REAL, danger_level REAL, quality TEXT DEFAULT 'good');
    CREATE TABLE IF NOT EXISTS rainfall_grid (ts TEXT, x INTEGER, y INTEGER, mmhr REAL,
        PRIMARY KEY (ts, x, y));
    CREATE TABLE IF NOT EXISTS risk_scores (ts TEXT, x INTEGER, y INTEGER,
        score REAL, factors TEXT, PRIMARY KEY (ts, x, y));
    CREATE TABLE IF NOT EXISTS inundation_pred (horizon_hr INTEGER, ts TEXT,
        x INTEGER, y INTEGER, depth_m REAL,
        PRIMARY KEY (ts, horizon_hr, x, y));
    CREATE TABLE IF NOT EXISTS dams (id INTEGER PRIMARY KEY, name TEXT,
        capacity_mcm REAL, max_level_m REAL, lon REAL, lat REAL);
    CREATE TABLE IF NOT EXISTS dam_alerts (id INTEGER PRIMARY KEY AUTOINCREMENT,
        dam_id INTEGER, ts TEXT, level_m REAL, outflow_cumecs REAL,
        breach_flag INTEGER, notes TEXT);
    CREATE TABLE IF NOT EXISTS shelters (id INTEGER PRIMARY KEY, name TEXT,
        capacity INTEGER, occupied INTEGER, lon REAL, lat REAL);
    CREATE TABLE IF NOT EXISTS resources (id INTEGER PRIMARY KEY, kind TEXT,
        base_name TEXT, count INTEGER, lon REAL, lat REAL);
    CREATE TABLE IF NOT EXISTS population_grid (x INTEGER, y INTEGER, pop INTEGER,
        PRIMARY KEY (x, y));
    CREATE TABLE IF NOT EXISTS mobile_aggs (ts TEXT, geohash TEXT, count INTEGER,
        PRIMARY KEY (ts, geohash));
    CREATE TABLE IF NOT EXISTS dispatch_plans (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, plan_json TEXT, score REAL);
    CREATE TABLE IF NOT EXISTS notifications (id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, channel TEXT, audience TEXT, message TEXT, severity TEXT, meta TEXT);
    CREATE TABLE IF NOT EXISTS grid_meta (key TEXT PRIMARY KEY, value TEXT);
    CREATE INDEX IF NOT EXISTS idx_rain_ts ON rainfall_obs(ts);
    CREATE INDEX IF NOT EXISTS idx_wl_ts ON water_level_obs(ts);
    """)
    conn.commit()
    return conn


# ══════════════════════════════════════════════════════════════════
# Load real/cached static data into SQLite
# ══════════════════════════════════════════════════════════════════

import geopandas as gpd
from shapely.geometry import Point


def setup_data():
    """Generate synthetic DEM (offline) and load fallback OSM layers into state DB."""
    import scripts.setup_real_data as s
    s.settings = settings
    # Ensure data dirs
    for d in ["raw/dem", "processed", "cache"]:
        (settings.data_dir / d).mkdir(parents=True, exist_ok=True)
    dem_path = settings.raw_dir / "dem" / "srtm_mandakini.tif"
    if not dem_path.exists():
        s._synthetic_dem(dem_path)
    for kind in ("rivers", "roads", "settlements", "pois", "admin"):
        out = settings.processed_dir / f"{kind}.geojson"
        if not out.exists():
            gdf = s._fallback_osm(kind)
            gdf.to_file(out, driver="GeoJSON")

    # Load into DB
    conn = get_conn()
    cur = conn.cursor()
    # Settlements
    gdf = gpd.read_file(settings.processed_dir / "settlements.geojson")
    cur.execute("DELETE FROM settlements")
    for _, r in gdf.iterrows():
        cur.execute(
            "INSERT OR REPLACE INTO settlements (osm_id, name, population, lon, lat) "
            "VALUES (?, ?, ?, ?, ?)",
            (int(r.osm_id) if r.osm_id else None, r.name, int(r.population or 0),
             r.geometry.x, r.geometry.y))
    # POIs
    gdf = gpd.read_file(settings.processed_dir / "pois.geojson")
    cur.execute("DELETE FROM pois")
    for _, r in gdf.iterrows():
        cur.execute(
            "INSERT OR REPLACE INTO pois (osm_id, name, amenity, category, lon, lat) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (int(r.osm_id) if r.osm_id else None, r.name, r.amenity, r.category,
             r.geometry.x, r.geometry.y))

    # Rivers/roads as GeoJSON strings
    for table, key in [("rivers","waterway"),("roads","highway")]:
        gdf = gpd.read_file(settings.processed_dir / f"{table}.geojson")
        cur.execute(f"DELETE FROM {table}")
        for _, r in gdf.iterrows():
            import json as _j
            gj = _j.dumps({"type":"Feature","geometry":r.geometry.__geo_interface__,
                           "properties":{"name":r.get("name","")}})
            cur.execute(
                f"INSERT INTO {table} (osm_id, name, {key}, geojson) VALUES (?, ?, ?, ?)",
                (int(r.osm_id) if r.osm_id else None, r.get("name",""), r.get(key,""), gj))

    # Now seed synthetic gauges / dams / shelters / resources / population
    import scripts.seed_synthetic_standalone as seed_mod
    seed_mod.seed_all(cur, settings, dem_path)

    # Grid meta
    import rasterio
    with rasterio.open(dem_path) as src:
        cur.execute(
            "INSERT OR REPLACE INTO grid_meta (key, value) VALUES (?, ?)",
            ("dem_bounds", json.dumps(list(src.bounds))))
        cur.execute(
            "INSERT OR REPLACE INTO grid_meta (key, value) VALUES (?, ?)",
            ("dem_shape", json.dumps([src.height, src.width])))
        cur.execute(
            "INSERT OR REPLACE INTO grid_meta (key, value) VALUES (?, ?)",
            ("dem_transform", json.dumps(list(src.transform)[:6])))

    conn.commit()


# ══════════════════════════════════════════════════════════════════
# Standalone simulator, risk, spread, optimizer (reuse existing modules)
# ══════════════════════════════════════════════════════════════════

def run_simulators():
    """Simulator thread — injects rain/water/mobile data into SQLite and bus."""
    from backend.app.ingestion import simulators_standalone as sim
    sim.run(bus, get_conn, settings)


def run_risk_worker():
    from backend.app.services import risk_worker_standalone as rw
    rw.run(bus, get_conn, settings)


def run_spread_worker():
    from backend.app.services import spread_worker_standalone as sw
    sw.run(bus, get_conn, settings)


def run_optimizer_worker():
    from backend.app.services import optimizer_worker_standalone as ow
    ow.run(bus, get_conn, settings)


# ══════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════

app = FastAPI(title="Flood Response — Standalone Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- SSE events ---
@app.get("/events")
async def events(request: Request):
    async def gen():
        # Subscribe to all event channels
        yield {"event":"hello","data":json.dumps({"ok":True})}
        channels = ["events.rain","events.water","events.dam","events.risk",
                    "events.inundation","events.dispatch","events.alert",
                    "events.cloudburst","events.mobile"]
        # Use a single thread per request that polls bus.latest via a queue
        q = queue.Queue(maxsize=200)
        # Subscribe by polling in a background thread
        def poller():
            seen = {}
            for ch in channels:
                seen[ch] = None
            while True:
                for ch in channels:
                    latest = bus.latest(ch)
                    if latest and latest is not seen.get(ch):
                        seen[ch] = latest
                        try:
                            q.put_nowait((ch.replace("events.",""), latest))
                        except queue.Full:
                            pass
                time.sleep(0.3)
        t = threading.Thread(target=poller, daemon=True)
        t.start()
        while True:
            if await request.is_disconnected():
                break
            try:
                ev, payload = q.get(timeout=1.0)
                yield {"event": ev, "data": json.dumps(payload)}
            except queue.Empty:
                yield {"event":"ping","data":"{}"}
    return EventSourceResponse(gen())


from backend.app.api_standalone import register_routes
register_routes(app, bus, get_conn, settings)


# Serve the built dashboard
DASH_DIR = Path(__file__).resolve().parent / "dashboard" / "dist"
if DASH_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(DASH_DIR), html=True), name="ui")
    @app.get("/")
    def root_redir():
        return FileResponse(str(DASH_DIR / "index.html"))
else:
    @app.get("/")
    def root():
        return {"service":"Flood Response Standalone", "dashboard":"build with `cd dashboard && npm run build` or run dev server"}


def main():
    print("═"*70)
    print("  AI Cloudburst → Flash Flood Response System (Standalone Mode)")
    print("  Basin: Mandakini River, Kedarnath (Uttarakhand, India)")
    print("═"*70)
    print("\n[setup] Initializing database and loading terrain/geography...")
    init_db()
    setup_data()
    print("[setup] Data ready.\n")

    # Start worker threads
    threads = []
    for fn, name in [
        (run_simulators, "simulators"),
        (run_risk_worker, "risk"),
        (run_spread_worker, "spread"),
        (run_optimizer_worker, "optimizer"),
    ]:
        t = threading.Thread(target=fn, name=name, daemon=True)
        t.start()
        threads.append(t)
        print(f"[{name}] worker started")

    time.sleep(2)
    print("\n" + "═"*70)
    print("  ✅ System online")
    print("     Dashboard:  http://localhost:8000")
    print("     API docs:   http://localhost:8000/docs")
    print("     Trigger:    POST /api/scenario/trigger {\"scenario\":\"cloudburst\"}")
    print("═"*70)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
