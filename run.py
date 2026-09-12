#!/usr/bin/env python3
"""
run.py — Complete single-file runnable demo.

One command:  python run.py
Then open:    http://localhost:8000

Requires: pip install fastapi uvicorn numpy geopandas shapely rasterio
requests scipy ortools sse-starlette pydantic

Runs entirely offline with synthetic DEM + fallback OSM (no Postgres/Redis).
Background threads simulate feeds + risk + spread + optimization.
A map-first web dashboard is served on the same port with live SSE updates.
"""
from __future__ import annotations

import io
import json
import math
import queue
import random
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ── FastAPI / web ──
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import uvicorn

# ── Geo stack ──
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds, Affine
from scipy.ndimage import gaussian_filter, distance_transform_edt, uniform_filter
from shapely.geometry import LineString, Point, Polygon

# ── OR-Tools ──
from ortools.sat.python import cp_model
import httpx


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

BASIN = {
    "name": "Mandakini River Basin (Kedarnath, Uttarakhand)",
    # Bounding box tuned to frame Kedarnath (top) → Rudraprayag (bottom) with
    # the Mandakini valley centered. All 20 named settlements visible at zoom 10.
    "bbox": [78.93, 30.49, 79.20, 30.77],
    # Visual center of the valley corridor, slightly south of Kedarnath
    "center": [79.060, 30.640],
}
GRID_RES_DEG = 10.0 / 3600.0   # ~300 m
SEED = 42
RNG = np.random.default_rng(SEED)
MANNING_N = 0.045

DATA_DIR = Path(__file__).resolve().parent / "data"
for d in ["raw/dem", "processed", "cache"]:
    (DATA_DIR / d).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# EVENT BUS (in-memory, replaces Redis)
# ══════════════════════════════════════════════════════════════════

class Bus:
    def __init__(self):
        self._latest: dict[str, dict] = {}
        self._kvs: dict[str, str] = {}
        self._listeners: list[queue.Queue] = []
        self._lock = threading.Lock()

    def publish(self, channel: str, payload: dict):
        payload = dict(payload)
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        self._latest[channel] = payload
        with self._lock:
            listeners = list(self._listeners)
        for q in listeners:
            try:
                q.put_nowait((channel, payload))
            except queue.Full:
                pass

    def latest(self, channel: str) -> dict | None:
        return self._latest.get(channel)

    def set(self, key: str, val: str): self._kvs[key] = val
    def get(self, key: str) -> str | None: return self._kvs.get(key)

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=500)
        with self._lock:
            self._listeners.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            self._listeners = [l for l in self._listeners if l is not q]


bus = Bus()


# ══════════════════════════════════════════════════════════════════
# TERRAIN (REAL-coordinate synthetic DEM + fallback OSM)
# ══════════════════════════════════════════════════════════════════

def make_synthetic_dem() -> tuple[np.ndarray, Affine]:
    """Generate a synthetic-but-plausible DEM of the Mandakini valley with
    a carved U-shaped river channel from Kedarnath to Rudraprayag."""
    west, south, east, north = BASIN["bbox"]
    w = int((east - west) / GRID_RES_DEG)
    h = int((north - south) / GRID_RES_DEG)
    xs = np.linspace(west, east, w)
    ys = np.linspace(south, north, h)
    XX, YY = np.meshgrid(xs, ys)

    lat_norm = (YY - south) / (north - south)
    elev = 700 + 5800 * lat_norm**1.4
    np.random.seed(SEED)
    noise = np.random.normal(0, 200, XX.shape)
    elev += gaussian_filter(noise, sigma=4)

    # Carve Mandakini valley
    for i in range(h):
        lat = south + i * GRID_RES_DEG
        if lat > 30.72:
            clon = 79.065 - 0.08 * (lat - 30.72) / (north - 30.72 + 1e-9)
        else:
            clon = 79.065 - 0.15 * (30.72 - lat) / (30.72 - south + 1e-9)
        dist_deg = np.abs(XX[i, :] - clon)
        elev[i, :] -= 1200 * np.exp(-(dist_deg / 0.015)**2)

    # Tributaries
    tribs = [(79.12,30.65,79.08,30.70),(79.00,30.68,79.05,30.71),
             (79.18,30.58,79.10,30.65),(78.98,30.55,79.02,30.63)]
    for (x1,y1,x2,y2) in tribs:
        for t in np.linspace(0,1,30):
            clon = x1 + t*(x2-x1); clat = y1 + t*(y2-y1)
            d = np.sqrt((XX-clon)**2 + (YY-clat)**2)
            elev -= 600 * np.exp(-(d/0.008)**2) * (1-t)

    elev = np.clip(elev, 580, 7800).astype(np.float32)
    transform = from_bounds(west, south, east, north, w, h)
    print(f"  DEM grid: {w} x {h} pixels ({GRID_RES_DEG*3600:.1f} arcsec = "
          f"~{GRID_RES_DEG*111320*math.cos(math.radians(30.7)):.0f} m per pixel)")
    # Save GeoTIFF
    dem_path = DATA_DIR / "raw" / "dem" / "srtm_mandakini.tif"
    profile = {"driver":"GTiff","dtype":"float32","height":h,"width":w,
               "count":1,"crs":"EPSG:4326","transform":transform,"nodata":-9999}
    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(elev, 1)
    return elev, transform


def fallback_osm(kind: str) -> gpd.GeoDataFrame:
    w, s, e, n = BASIN["bbox"]
    feats = []
    if kind == "rivers":
        feats.append({"osm_id":1,"name":"Mandakini","waterway":"river",
            "geometry":LineString([(79.066,30.735),(79.064,30.71),(79.060,30.685),
                (79.055,30.66),(79.045,30.63),(79.030,30.60),(79.010,30.57),
                (78.985,30.54),(78.965,30.505)])})
        feats.append({"osm_id":2,"name":"Mandani Ganga","waterway":"river",
            "geometry":LineString([(79.14,30.68),(79.10,30.66),(79.075,30.645),(79.055,30.66)])})
        feats.append({"osm_id":3,"name":"Kali Ganga","waterway":"river",
            "geometry":LineString([(78.96,30.66),(79.00,30.62),(79.02,30.60)])})
    elif kind == "roads":
        nh109 = [(78.965,30.505),(78.98,30.53),(79.01,30.56),(79.035,30.59),
                 (79.055,30.62),(79.075,30.645),(79.09,30.67),(79.11,30.695),
                 (79.08,30.71),(79.066,30.735)]
        feats.append({"osm_id":100,"name":"NH109","highway":"primary","geometry":LineString(nh109)})
        feats.append({"osm_id":101,"name":"Ukhimath road","highway":"secondary",
            "geometry":LineString([(79.075,30.645),(79.10,30.62),(79.13,30.60)])})
        feats.append({"osm_id":102,"name":"Triyuginarayan road","highway":"tertiary",
            "geometry":LineString([(79.09,30.67),(79.13,30.68)])})
    elif kind == "settlements":
        places = [
            ("Kedarnath",79.066,30.735,800),("Gaurikund",79.080,30.710,1200),
            ("Rambara",79.073,30.700,100),("Garudiya Chatti",79.065,30.688,80),
            ("Bhiri",79.055,30.67,150),("Phata",79.090,30.670,800),
            ("Guptkashi",79.075,30.645,3500),("Ukhimath",79.130,30.600,3000),
            ("Kund",79.055,30.620,1500),("Agastyamuni",79.035,30.590,2500),
            ("Chandrapuri",79.010,30.570,1200),("Tilwara",78.990,30.550,900),
            ("Rudraprayag",78.965,30.505,12000),("Trijuginarayan",79.130,30.680,400),
            ("Kalimath",79.105,30.650,300),("Sersi",79.085,30.680,400)]
        for n,lo,la,p in places:
            feats.append({"osm_id":abs(hash(n))%10**9,"name":n,"population":p,"geometry":Point(lo,la)})
    elif kind == "pois":
        pois = [
            ("Gaurikund base hospital",79.080,30.710,"hospital","hospital"),
            ("Phata CHC",79.090,30.670,"hospital","hospital"),
            ("Guptkashi District Hospital",79.075,30.645,"hospital","hospital"),
            ("Agastyamuni CHC",79.035,30.590,"hospital","hospital"),
            ("Rudraprayag District Hospital",78.965,30.505,"hospital","hospital"),
            ("Ukhimath CHC",79.130,30.600,"hospital","hospital"),
            ("Govt Inter College Guptkashi",79.076,30.646,"school","school"),
            ("Police Station Phata",79.091,30.671,"police","police"),
            ("Police Station Guptkashi",79.076,30.646,"police","police"),
            ("Police Station Rudraprayag",78.966,30.507,"police","police"),
            ("SDRF Base Rudraprayag",78.968,30.508,"fire_station","fire")]
        for n,lo,la,a,c in pois:
            feats.append({"osm_id":abs(hash(n))%10**9,"name":n,"amenity":a,"category":c,"geometry":Point(lo,la)})
    return gpd.GeoDataFrame(feats, geometry="geometry", crs="EPSG:4326")


# ══════════════════════════════════════════════════════════════════
# HYDROLOGY LAYERS
# ══════════════════════════════════════════════════════════════════

def compute_hydrology(dem: np.ndarray, transform: Affine,
                      settlements: gpd.GeoDataFrame):
    H, W = dem.shape
    # Slope (degrees)
    lat_c = 30.73
    dy_m = abs(transform.e) * 111320.0
    dx_m = abs(transform.a) * 111320.0 * math.cos(math.radians(lat_c))
    dy, dx = np.gradient(dem, dy_m, dx_m)
    slp = np.degrees(np.arctan(np.sqrt(dx*dx + dy*dy)))

    # D8 flow direction
    fd = np.zeros_like(dem, dtype=np.uint8)
    elev = np.pad(dem, 1, mode="edge")
    dirmap = [(0,1,1),(1,1,2),(1,0,4),(1,-1,8),(0,-1,16),(-1,-1,32),(-1,0,64),(-1,1,128)]
    drops = []
    for di,dj,code in dirmap:
        shifted = elev[1+di:H+1+di,1+dj:W+1+dj]
        dist = math.sqrt(2) if di!=0 and dj!=0 else 1.0
        drops.append(((dem - shifted)/dist, code, di, dj))
    max_drop = np.zeros_like(dem)
    receivers = {}
    inflow = np.zeros_like(dem, dtype=np.int32)
    dirs = {1:(0,1),2:(1,1),4:(1,0),8:(1,-1),16:(0,-1),32:(-1,-1),64:(-1,0),128:(-1,1)}
    for drop,code,di,dj in drops:
        mask = drop > max_drop
        fd[mask] = code
        max_drop[mask] = drop[mask]
    for i in range(H):
        for j in range(W):
            code = fd[i,j]
            if code not in dirs: continue
            di,dj = dirs[code]
            ni,nj = i+di, j+dj
            if 0<=ni<H and 0<=nj<W:
                inflow[ni,nj] += 1
                receivers[(i,j)] = (ni,nj)

    # Flow accumulation via topological BFS
    acc = np.ones_like(dem, dtype=np.float64)
    from collections import deque
    q = deque((i,j) for i in range(H) for j in range(W) if inflow[i,j]==0)
    while q:
        i,j = q.popleft()
        if (i,j) not in receivers: continue
        ni,nj = receivers[(i,j)]
        acc[ni,nj] += acc[i,j]
        inflow[ni,nj] -= 1
        if inflow[ni,nj] == 0:
            q.append((ni,nj))

    # Rasterize rivers (LineStrings from OSM fallback)
    rivers_gdf = fallback_osm("rivers")
    from rasterio import features as rast_feat
    river_mask = rast_feat.rasterize(
        [(g,1) for g in rivers_gdf.geometry], out_shape=(H,W),
        transform=transform, fill=0, dtype=np.uint8, all_touched=True)
    dist_river_px = distance_transform_edt(1 - river_mask)
    px_m = (dx_m + dy_m)/2.0
    dist_river = dist_river_px * px_m
    drain_dens = uniform_filter(river_mask.astype(np.float64), size=15)

    return {"dem":dem,"slope":slp,"flowdir":fd,"flowacc":acc,"rivers":river_mask,
            "dist_to_river":dist_river,"drainage_density":drain_dens,
            "transform":transform,"H":H,"W":W,"pixel_m":px_m}


def manning_depth(flowacc, slope, n=MANNING_N, pixel_m=300.0, rain_excess_mhr=0.05):
    contrib_m2 = flowacc * pixel_m * pixel_m
    Q = rain_excess_mhr/3600.0 * contrib_m2
    safe_slope = np.maximum(np.sin(np.radians(np.maximum(slope,0.001))),1e-4)
    d = np.maximum(0.0, (Q*n/(pixel_m*np.sqrt(safe_slope)))**(3/5))
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


# ══════════════════════════════════════════════════════════════════
# SIMULATOR, RISK, SPREAD, OPTIMIZER
# ══════════════════════════════════════════════════════════════════

class State:
    def __init__(self, hydro, settlements, pois, shelters, resources):
        self.hydro = hydro
        self.settlements = settlements
        self.pois = pois
        self.shelters = shelters
        self.resources = resources
        # Gauges: (code, name, type, lon, lat, danger_level, level_m, mmhr)
        self.gauges = [
            ("R-KED","Kedarnath rain","rain",79.066,30.735,None,0.8,0.0),
            ("R-GAU","Gaurikund rain","rain",79.080,30.710,None,1.2,0.0),
            ("R-PHA","Phata rain","rain",79.090,30.670,None,1.5,0.0),
            ("R-GUP","Guptkashi rain","rain",79.075,30.645,None,2.0,0.0),
            ("R-AKA","Agastyamuni rain","rain",79.035,30.590,None,3.0,0.0),
            ("R-RUD","Rudraprayag rain","rain",78.965,30.505,None,4.0,0.0),
            ("W-GAU","Gaurikund WL","water",79.080,30.708,1.8,None,1.0),
            ("W-PHA","Phata WL","water",79.088,30.672,2.2,None,1.0),
            ("W-GUP","Guptkashi WL","water",79.073,30.647,2.8,None,1.0),
            ("W-AKA","Agastyamuni WL","water",79.033,30.592,3.5,None,1.0),
            ("W-RUD","Rudraprayag WL","water",78.967,30.508,5.0,None,1.2),
        ]
        self.rain_grid = np.zeros((hydro["H"], hydro["W"]), dtype=np.float32)
        self.wl_state = {g[0]: 1.0 for g in self.gauges if g[2]=="water"}
        self.antecedent = {g[0]: 0.0 for g in self.gauges if g[2] in ("rain","water")}
        self.risk = None
        self.inundation = {}  # horizon -> depth_m array
        self.dispatch_plan = None
        self.notifications = []
        self.cb_active = False
        self.breach_active = False
        self.cb_t0 = None
        self.cb_pos = (79.068, 30.735)
        self.cb_age_hr = -1


def sim_cloudburst_rain(t_hr, XX, YY):
    """Gaussian rain cell drifting SSW at ~4 km/hr. Lasts ~12 sim hours."""
    if t_hr < 0 or t_hr > 12:
        return np.zeros_like(XX)
    clon = 79.068 - 0.013*t_hr
    clat = 30.735 - 0.030*t_hr
    # Peaks around hour 2-3, sustains moderate rain until hour 8
    env = math.exp(-((t_hr - 2.5)**2)/(2*3.5**2))
    peak = 220.0 * env
    sigma = 0.014  # broader cell
    return peak * np.exp(-((XX-clon)**2 + (YY-clat)**2)/(2*sigma**2))


def simulator_loop(state: State, stop_event: threading.Event):
    hydro = state.hydro
    H, W, tr = hydro["H"], hydro["W"], hydro["transform"]
    cols = np.arange(W); rows = np.arange(H)
    CC, RR = np.meshgrid(cols, rows)
    XX = tr.c + (CC+0.5)*tr.a + (RR+0.5)*tr.b
    YY = tr.f + (CC+0.5)*tr.d + (RR+0.5)*tr.e

    sim_hr = 0.0
    tick_dt_sim_hr = 1.5/60.0  # 1.5 min sim per 0.5 sec tick → 3 min/s → event lasts ~4 min real
    while not stop_event.is_set():
        t0 = time.time()
        sim_hr += tick_dt_sim_hr

        active = state.cb_active
        if active:
            if state.cb_t0 is None:
                state.cb_t0 = sim_hr  # simulated-hour timestamp
            state.cb_age_hr = sim_hr - state.cb_t0
            if state.cb_age_hr > 13.0:
                state.cb_active = False
                state.cb_age_hr = -1
                state.cb_t0 = None
                continue

        # Rain grid
        base = np.maximum(0.0, 1.0 + RNG.normal(0,0.3,(H,W)))
        if active and state.cb_age_hr >= 0:
            base += sim_cloudburst_rain(state.cb_age_hr, XX, YY)
        state.rain_grid = base

        # Update gauge values
        new_gauges = []
        for g in state.gauges:
            code,name,typ,lon,lat,dl,_lvl,_mm = g
            if typ == "rain":
                # Find pixel
                c = int((lon-tr.c)/tr.a); r = int((lat-tr.f)/tr.e)
                c = max(0,min(W-1,c)); r = max(0,min(H-1,r))
                mmhr = float(base[r,c])
                state.antecedent[code] = state.antecedent.get(code,0) + mmhr*(10.0/60.0)
                state.antecedent[code] *= math.exp(-(10.0/60.0)/24.0)
                new_gauges.append((code,name,typ,lon,lat,dl,_lvl,mmhr))
                bus.publish("events.rain", {"code":code,"name":name,"lon":lon,"lat":lat,"mmhr":round(mmhr,2)})
            else:
                new_gauges.append(g)

        # Water levels (routed wave): use RAIN RATE (not cumulative) at nearest upstream gauge
        wl_updates = {}
        # Sort gauges: upper-basin first
        wl_codes = sorted([g for g in new_gauges if g[2]=="water"],
                          key=lambda g: -g[4])
        for i,g in enumerate(wl_codes):
            code,name,typ,lon,lat,dl,_lvl,_mm = g
            # Sample rain from the rain grid directly at gauge pixel + smoothed surrounding
            c = int((lon-tr.c)/tr.a); r = int((lat-tr.f)/tr.e)
            c = max(2,min(W-3,c)); r = max(2,min(H-3,r))
            nearest_rain = float(state.rain_grid[r-2:r+3, c-2:c+3].max())
            # Fall back to any upstream rain gauge
            for rg in new_gauges:
                if rg[2] != "rain": continue
                if rg[4] >= lat - 0.05:
                    dd = math.sqrt((rg[3]-lon)**2 + (rg[4]-lat)**2)
                    if dd < 0.15:
                        nearest_rain = max(nearest_rain, rg[7] or 0)
            # Distance-based lag from upstream gauge: rough wave travel
            if i == 0:
                offset_hr = 0.3
            else:
                prev = wl_codes[i-1]
                d_km = haversine_km(prev[3], prev[4], lon, lat)
                offset_hr = d_km / 15.0  # 15 km/hr wave speed
            # Target rise proportional to rain intensity above 10 mm/hr (runoff)
            excess = max(0, nearest_rain - 10.0)
            # Target level: baseline 1 m + rise scaled by rain, with routing delay
            rise = 0.0
            if active and state.cb_age_hr is not None and state.cb_age_hr >= 0:
                t_ev = state.cb_age_hr - offset_hr
                if t_ev > 0:
                    # Bell-shaped flood wave at each station (rise + recession)
                    rise = min(dl*1.2, 0.020*excess*math.exp(-((t_ev-1.2)**2)/3.0))
            if state.breach_active and state.cb_age_hr is not None:
                t_ev = (state.cb_age_hr or 0) - offset_hr
                if t_ev > 0:
                    rise = max(rise, min(dl*2.0, 5.0*math.exp(-t_ev/2.0)))
            target = 1.0 + 0.05*math.sin(sim_hr*0.2) + rise
            # Smooth approach (slow decay back to baseline)
            cur = state.wl_state.get(code, 1.0)
            alpha = 0.15 if target > cur else 0.05
            new_val = max(0.8, cur + alpha*(target-cur) + RNG.normal(0,0.01))
            new_val = min(new_val, (dl or 3.0)*2.5)  # cap at 2.5x danger level
            state.wl_state[code] = new_val
            wl_updates[code] = new_val
            bus.publish("events.water",{
                "code":code,"name":name,"lon":lon,"lat":lat,
                "level_m":round(new_val,2),"danger_m":dl})

        # Update gauge list with live levels/mmhr
        final = []
        for g in new_gauges:
            code,name,typ,lon,lat,dl,_l,_m = g
            if typ == "water":
                lvl = wl_updates.get(code, state.wl_state.get(code, 1.0))
                final.append((code,name,typ,lon,lat,dl,lvl,None))
            else:
                final.append(g)
        state.gauges = final

        # Mobile pings near settlements
        mobile = []
        for _, s in state.settlements.iterrows():
            gh = _geohash(s.geometry.x, s.geometry.y, 6)
            base_cnt = int((s.population or 500)*0.05)
            cnt = int(RNG.poisson(base_cnt))
            if active and state.cb_age_hr and 0 < state.cb_age_hr < 5:
                cnt = int(cnt*(1.0 + 0.5*math.exp(-state.cb_age_hr)))
            mobile.append({"geohash":gh,"count":cnt,"lon":s.geometry.x,"lat":s.geometry.y})
        bus.publish("events.mobile",{"cells":mobile[:10]})

        # Cycle sleep — short tick for near-real-time feel
        elapsed = time.time()-t0
        time.sleep(max(0.2, 0.5 - elapsed))


def compute_risk(state: State):
    hydro = state.hydro
    H, W = hydro["H"], hydro["W"]
    rain = state.rain_grid
    rain_smooth = gaussian_filter(rain, sigma=3)
    rain_score = np.clip(rain_smooth/100.0*30, 0, 30)
    ant_score = np.clip(rain_smooth/50.0*5, 0, 10)
    slp = hydro["slope"]
    slope_score = np.where(slp<5,12,np.where(slp<15,8,np.where(slp<25,4,1)))
    acc = hydro["flowacc"]
    acc_score = np.clip(np.log1p(acc)/math.log(acc.size)*15,0,15)
    d2r = hydro["dist_to_river"]
    dist_score = np.where(d2r<100,15,np.where(d2r<300,10,np.where(d2r<800,5,1)))
    any_danger = any((g[6] or 0) >= (g[5] or 99) for g in state.gauges if g[2]=="water")
    any_warn = any((g[6] or 0) >= 0.8*(g[5] or 99) for g in state.gauges if g[2]=="water")
    wl_bonus = 10 if any_danger else (5 if any_warn else 0)
    breach_score = np.zeros_like(rain)
    if state.breach_active:
        breach_score = np.where((acc>np.percentile(acc,90)) & (slp<20),20,
                        np.where(acc>np.percentile(acc,75),10,0))
    risk = np.clip(rain_score+ant_score+slope_score+acc_score+dist_score+wl_bonus+breach_score,0,100).astype(np.float32)
    state.risk = risk

    top = []
    flat_idx = np.argsort(risk.ravel())[-10:][::-1]
    tr = hydro["transform"]
    for idx in flat_idx:
        r,c = divmod(int(idx), W)
        s = float(risk[r,c])
        if s < 20: break
        lon = tr.c + (c+0.5)*tr.a; lat = tr.f + (r+0.5)*tr.e
        top.append({"x":int(c),"y":int(r),"score":round(s,1),"lon":round(lon,4),"lat":round(lat,4)})

    bus.publish("events.risk",{
        "max_score": float(risk.max()),
        "mean_score": float(risk.mean()),
        "cells_high": int((risk>70).sum()),
        "cells_moderate": int(((risk>40)&(risk<=70)).sum()),
        "top": top,
    })
    if risk.max() >= 80:
        add_notification(state,"critical","authority","SDMA/DDMA",
            f"EXTREME flood risk (score {risk.max():.0f}) — evacuation advisory.")
    elif risk.max() >= 60:
        add_notification(state,"warn","authority","SDMA/DDMA",
            f"HIGH flood risk (score {risk.max():.0f}) — prepare evacuations.")
    return risk


def simulate_spread(state: State):
    hydro = state.hydro
    H, W = hydro["H"], hydro["W"]; tr = hydro["transform"]
    max_rain = float(state.rain_grid.max())
    rain_excess = min(200.0, max_rain) if (state.cb_active or state.breach_active) else min(5.0, max_rain)
    depth_base = manning_depth(hydro["flowacc"], hydro["slope"], MANNING_N,
                                hydro["pixel_m"], rain_excess_mhr=max(0.005,rain_excess/1000.0))
    scale = np.clip(rain_excess/50.0,0.0,5.0)

    # Velocity + travel time per pixel
    n = MANNING_N
    S = np.maximum(np.sin(np.radians(np.maximum(hydro["slope"],0.001))),1e-4)
    v = (1/n)*np.maximum(depth_base,0.05)**(2/3)*np.sqrt(S)
    v = np.clip(v,0.5,8.0)
    t_px = hydro["pixel_m"]/v/3600.0  # hours per pixel

    # Seed ETA from upper-basin high-acc pixels
    rows_lat = tr.f + (np.arange(H)+0.5)*tr.e
    INF = 1e9
    eta = np.full((H,W), INF, dtype=np.float32)
    seed_mask = (hydro["flowacc"] > np.percentile(hydro["flowacc"],99)) & (rows_lat[:,None] > 30.68)
    if state.breach_active:
        dc = int((79.062-tr.c)/tr.a); dr = int((30.750-tr.f)/tr.e)
        if 0<=dc<W and 0<=dr<H:
            seed_mask[dr,dc] = True
    ys,xs = np.where(seed_mask)
    eta[ys,xs] = 0.0

    # Fast marching iterations
    for _ in range(80):
        new_eta = eta.copy()
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            shifted = np.pad(eta,1,constant_values=INF)[1+di:H+1+di,1+dj:W+1+dj]
            df = math.sqrt(2) if di!=0 and dj!=0 else 1.0
            new_eta = np.minimum(new_eta, shifted+t_px*df)
        if np.allclose(new_eta, eta, atol=0.01):
            break
        eta = new_eta

    results = {}
    for hh in (1,3,6,12):
        wetted = eta < hh
        tsa = np.clip(hh-eta,0,hh)
        df = np.clip(tsa/max(1,hh/2),0.2,1.0)
        depth_h = depth_base*scale*df
        if state.breach_active:
            dc = int((79.062-tr.c)/tr.a); dr = int((30.750-tr.f)/tr.e)
            if 0<=dc<W and 0<=dr<H:
                Q_t = 1500.0*math.exp(-max(0,(state.cb_age_hr or 0)+hh - 0.5)/2.0)
                breach_amp = 8.0*Q_t/1500.0
                rr,cc = np.ogrid[0:H,0:W]
                bd = np.sqrt((rr-dr)**2+(cc-dc)**2)
                breach_mask = (hydro["flowacc"]>np.percentile(hydro["flowacc"],90)) & (bd<hh*15)
                depth_h = np.where(breach_mask, np.maximum(depth_h,breach_amp*np.exp(-bd/20)),depth_h)
        depth_h = np.where(wetted, depth_h, 0.0)
        depth_h = np.where(hydro["dem"]<600,0.0,depth_h)
        results[hh] = depth_h.astype(np.float32)
    state.inundation = results

    bus.publish("events.inundation",{
        "horizons":[1,3,6,12],
        "max_depths":{str(h):float(d.max()) for h,d in results.items()},
    })
    return results


def haversine_km(lon1,lat1,lon2,lat2):
    R=6371.0
    p1=math.radians(lat1);p2=math.radians(lat2)
    dp=math.radians(lat2-lat1);dl=math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def tt_minutes(lon1,lat1,lon2,lat2):
    return haversine_km(lon1,lat1,lon2,lat2)*1.3/25.0*60.0


def build_zones_and_dispatch(state: State):
    """Identify at-risk settlements and run OR-Tools optimizer."""
    hydro = state.hydro
    H, W, tr = hydro["H"], hydro["W"], hydro["transform"]
    depth12 = state.inundation.get(12)
    if depth12 is None:
        return

    # Find at-risk settlements: near deep water
    zones = []
    eta_per = {}
    zid = 0
    for _, s in state.settlements.iterrows():
        slon,slat = s.geometry.x, s.geometry.y
        c = int((slon-tr.c)/tr.a); r = int((slat-tr.f)/tr.e)
        if not (0<=c<W and 0<=r<H): continue
        r0,r1 = max(0,r-6),min(H,r+6); c0,c1 = max(0,c-6),min(W,c+6)
        window = depth12[r0:r1,c0:c1]
        near = window.max() > 0.3
        eta_h = 12
        for hh in (1,3,6,12):
            w = state.inundation[hh][r0:r1,c0:c1]
            if w.max() > 0.3:
                eta_h = hh; break
        pop = int(s.population or 500)
        # In non-event mode, show planning baseline (just largest towns)
        if not (state.cb_active or state.breach_active) and not near:
            continue
        zones.append({"id":zid,"name":s["name"],"lon":slon,"lat":slat,
                      "population":pop,"flood_eta_h":eta_h})
        eta_per[zid] = eta_h
        zid += 1

    # If no active event, plan for upper basin as demo
    if not zones:
        for _, s in state.settlements.head(6).iterrows():
            zones.append({"id":zid,"name":s["name"],"lon":s.geometry.x,
                         "lat":s.geometry.y,"population":int((s.population or 500)//4),
                         "flood_eta_h":12})
            eta_per[zid] = 12
            zid += 1

    # Build resource list from shelters+resources (use hardcoded plausible)
    resources = []; rid = 1
    for kind,name,cnt,lon,lat in [
        ("team","SDRF Rudraprayag",6,78.968,30.508),
        ("team","ITBP Phata",4,79.092,30.672),
        ("team","SDRF Guptkashi",3,79.077,30.645),
        ("team","Civil Defence Ukhimath",2,79.130,30.601),
        ("boat","Rudraprayag boats",12,78.966,30.506),
        ("boat","Phata boats",6,79.090,30.671),
        ("boat","Agastyamuni boats",5,79.036,30.591),
        ("boat","Guptkashi boats",4,79.074,30.646),
        ("ambulance","Rudraprayag ambulance",8,78.965,30.505),
        ("ambulance","Guptkashi ambulance",4,79.075,30.645),
        ("ambulance","Agastyamuni ambulance",3,79.035,30.590),
        ("ambulance","Phata ambulance",2,79.090,30.670),
        ("supplies","Rudraprayag warehouse",10000,78.970,30.510),
    ]:
        resources.append({"id":rid,"kind":kind,"name":name,"count":cnt,"lon":lon,"lat":lat})
        rid += 1
    shelters = [
        {"id":1,"name":"Rudraprayag stadium","capacity":5000,"lon":78.965,"lat":30.505},
        {"id":2,"name":"Guptkashi town hall","capacity":3000,"lon":79.075,"lat":30.645},
        {"id":3,"name":"Phata shelter","capacity":2000,"lon":79.090,"lat":30.670},
        {"id":4,"name":"Agastyamuni college","capacity":2500,"lon":79.035,"lat":30.590},
        {"id":5,"name":"Ukhimath school","capacity":2500,"lon":79.130,"lat":30.600},
    ]

    # Build demands
    demands = []
    for z in zones:
        pop = z["population"]; fh = z["flood_eta_h"]
        urgency = 1.0 if fh<=1 else (0.7 if fh<=3 else 0.4)
        demands.append({
            "zone_id":z["id"],"zone_name":z["name"],"lon":z["lon"],"lat":z["lat"],
            "flood_eta_h":fh,"urgency":urgency,
            "teams_needed":max(1,math.ceil(pop/500*urgency)),
            "boats_needed":max(1,math.ceil(pop/300)) if fh<=3 else max(0,math.ceil(pop/800)),
            "ambulances_needed":max(0,math.ceil(pop/1500)),
            "shelter_needed":math.ceil(pop*0.6),
        })

    # ── Solve CP-SAT ──
    model = cp_model.CpModel()
    x = {}
    travel_times = {}
    MAX_INT = 10_000_000
    for r in resources:
        for z in demands:
            tt = tt_minutes(r["lon"],r["lat"],z["lon"],z["lat"])
            cap = r["count"]
            var = model.NewIntVar(0,cap,f"x_{r['id']}_{z['zone_id']}")
            x[(r["id"],z["zone_id"])] = var
            travel_times[(r["id"],z["zone_id"])] = tt
    for r in resources:
        model.Add(sum(x[(r["id"],z["zone_id"])] for z in demands) <= r["count"])

    y = {}
    for s in shelters:
        for z in demands:
            var = model.NewIntVar(0,MAX_INT,f"y_{s['id']}_{z['zone_id']}")
            y[(s["id"],z["zone_id"])] = var
        model.Add(sum(y[(s["id"],z["zone_id"])] for z in demands) <= s["capacity"])

    unmet_teams={}; unmet_boats={}; unmet_amb={}; unmet_shelter={}
    for z in demands:
        ut = model.NewIntVar(0,z["teams_needed"],f"ut_{z['zone_id']}")
        ub = model.NewIntVar(0,z["boats_needed"],f"ub_{z['zone_id']}")
        ua = model.NewIntVar(0,z["ambulances_needed"],f"ua_{z['zone_id']}")
        us = model.NewIntVar(0,z["shelter_needed"],f"us_{z['zone_id']}")
        at_t = sum(x[(r["id"],z["zone_id"])] for r in resources if r["kind"]=="team")
        at_b = sum(x[(r["id"],z["zone_id"])] for r in resources if r["kind"]=="boat")
        at_a = sum(x[(r["id"],z["zone_id"])] for r in resources if r["kind"]=="ambulance")
        model.Add(at_t+ut==z["teams_needed"])
        model.Add(at_b+ub==z["boats_needed"])
        model.Add(at_a+ua==z["ambulances_needed"])
        model.Add(sum(y[(s["id"],z["zone_id"])] for s in shelters)+us==z["shelter_needed"])
        unmet_teams[z["zone_id"]]=ut; unmet_boats[z["zone_id"]]=ub
        unmet_amb[z["zone_id"]]=ua; unmet_shelter[z["zone_id"]]=us

    total_travel = []
    for r in resources:
        for z in demands:
            total_travel.append(int(travel_times[(r["id"],z["zone_id"])]*2)*x[(r["id"],z["zone_id"])])
    penalty_terms = []
    for z in demands:
        u = 3 if z["flood_eta_h"]<=1 else (2 if z["flood_eta_h"]<=3 else 1)
        penalty_terms.append(1000*u*unmet_teams[z["zone_id"]])
        penalty_terms.append(1000*u*unmet_boats[z["zone_id"]])
        penalty_terms.append(1000*u*unmet_amb[z["zone_id"]])
        penalty_terms.append(300*u*unmet_shelter[z["zone_id"]])
    model.Minimize(sum(total_travel)+sum(penalty_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3.0
    status = solver.Solve(model)

    plan = {"assignments":[],"shelter_assignments":[],"shortfalls":[],
            "zones":demands,"solver_status":solver.StatusName(status)}

    # Greedy baseline
    res_left = {r["id"]:r["count"] for r in resources}
    base_assn = []
    base_short = 0
    for z in sorted(demands, key=lambda z:z["flood_eta_h"]):
        for kind,key in [("team","teams_needed"),("boat","boats_needed"),("ambulance","ambulances_needed")]:
            need = z[key]
            avail = sorted([r for r in resources if r["kind"]==kind and res_left[r["id"]]>0],
                           key=lambda r: tt_minutes(r["lon"],r["lat"],z["lon"],z["lat"]))
            rem = need
            for r in avail:
                gv = min(rem, res_left[r["id"]])
                if gv>0:
                    base_assn.append(tt_minutes(r["lon"],r["lat"],z["lon"],z["lat"]))
                    res_left[r["id"]] -= gv; rem -= gv
                if rem==0: break
            base_short += rem
    base_avg = float(np.mean(base_assn)) if base_assn else 0.0

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for r in resources:
            for z in demands:
                val = solver.Value(x[(r["id"],z["zone_id"])])
                if val>0:
                    plan["assignments"].append({
                        "resource_kind":r["kind"],"resource_name":r["name"],
                        "resource_lon":r["lon"],"resource_lat":r["lat"],
                        "zone_name":z["zone_name"],"zone_lon":z["lon"],"zone_lat":z["lat"],
                        "units":val,"travel_minutes":round(travel_times[(r["id"],z["zone_id"])],1)})
        for s in shelters:
            for z in demands:
                val = solver.Value(y[(s["id"],z["zone_id"])])
                if val>0:
                    plan["shelter_assignments"].append({
                        "shelter_name":s["name"],"shelter_lon":s["lon"],"shelter_lat":s["lat"],
                        "zone_name":z["zone_name"],"people":val})
        for z in demands:
            ut=solver.Value(unmet_teams[z["zone_id"]]); ub=solver.Value(unmet_boats[z["zone_id"]])
            ua=solver.Value(unmet_amb[z["zone_id"]]); ush=solver.Value(unmet_shelter[z["zone_id"]])
            if ut+ub+ua+ush>0:
                plan["shortfalls"].append({
                    "zone_name":z["zone_name"],"teams_short":ut,"boats_short":ub,
                    "ambulances_short":ua,"shelter_short":ush})

    avg_tt = float(np.mean([a["travel_minutes"] for a in plan["assignments"]])) if plan["assignments"] else 0
    plan["avg_travel_minutes"] = round(avg_tt,1)
    plan["baseline_avg_minutes"] = round(base_avg,1)
    plan["baseline_shortfall"] = base_short
    state.dispatch_plan = plan

    bus.publish("events.dispatch",{
        "avg_travel_minutes":round(avg_tt,1),
        "num_assignments":len(plan["assignments"]),
        "num_shortfalls":len(plan["shortfalls"]),
        "baseline_avg_minutes":round(base_avg,1),
    })

    # Shortfall notifications
    for sf in plan["shortfalls"][:3]:
        add_notification(state,"critical","authority","SDMA/NDMA",
            f"Shortfall in {sf['zone_name']}: +{sf['teams_short']} teams, +{sf['boats_short']} boats, +{sf['ambulances_short']} ambulances — request mutual aid.")
    for z in demands[:6]:
        if z["flood_eta_h"]<=3:
            add_notification(state,"critical" if z["flood_eta_h"]<=1 else "warn",
                "cell_broadcast", f"zone:{z['zone_name']}",
                f"FLOOD WARNING: {z['zone_name']} expected within {z['flood_eta_h']}h — move to higher ground.")


def add_notification(state, severity, channel, audience, message):
    n = {"ts":datetime.now(timezone.utc).isoformat(),"severity":severity,
         "channel":channel,"audience":audience,"message":message}
    state.notifications.insert(0,n)
    state.notifications = state.notifications[:50]
    bus.publish("events.alert",n)


def worker_loop(state, stop_event):
    """Risk → spread → optimize cycle."""
    last_risk = 0; last_spread = 0; last_opt = 0
    while not stop_event.is_set():
        now = time.time()
        if now - last_risk >= 3:
            compute_risk(state); last_risk = now
        if now - last_spread >= 8:
            simulate_spread(state); last_spread = now
        if now - last_opt >= 15 and state.inundation:
            try:
                build_zones_and_dispatch(state)
            except Exception as e:
                import traceback; traceback.print_exc()
            last_opt = now
        time.sleep(1)


def _geohash(lon,lat,precision=6):
    _B="0123456789bcdefghjkmnpqrstuvwxyz"
    latR=[-90.0,90.0]; lonR=[-180.0,180.0]
    gh=""; bit=0; ch=0; even=True
    while len(gh)<precision:
        if even:
            mid=(lonR[0]+lonR[1])/2
            if lon>mid: ch|=(1<<(4-bit)); lonR[0]=mid
            else: lonR[1]=mid
        else:
            mid=(latR[0]+latR[1])/2
            if lat>mid: ch|=(1<<(4-bit)); latR[0]=mid
            else: latR[1]=mid
        even=not even; bit+=1
        if bit==5: gh+=_B[ch]; bit=0; ch=0
    return gh



# ───────────────────────────────────────────────────────────────────
# FastAPI APP
# ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Cloudburst → Flash Flood Response (Standalone)")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from fastapi import Request
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ───────────────────────────────────────────────────────────────────
# TILE PROXY (proxies free OSM/CARTO/ESRI tiles through the server with
# a proper User-Agent so the browser never hits tile servers directly —
# avoids UA blocks by OSM/CARTO/OpenTopoMap).
# ───────────────────────────────────────────────────────────────────

TILE_PROVIDERS = {
    "streets": [
        "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://b.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "https://c.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
    ],
    "topo": [
        "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        "https://b.tile.opentopomap.org/{z}/{x}/{y}.png",
        "https://c.tile.opentopomap.org/{z}/{x}/{y}.png",
    ],
    "esri_topo": [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    ],
    "esri_img": [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    ],
    "osm": [
        "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
    ],
}

_client = None
def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            headers={"User-Agent": "FloodResponseDemo/1.0 (research; contact: demo@example.com)"},
            timeout=httpx.Timeout(8.0, connect=4.0),
            follow_redirects=True,
            http2=False,
        )
    return _client

@app.get("/tiles/{provider}/{z}/{x}/{y}.png")
async def tile_proxy(provider: str, z: int, x: int, y: int):
    urls = TILE_PROVIDERS.get(provider)
    if not urls:
        return Response(content=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc\xff\xff?\x03\x00\x08\xfc\x02\xfe\xa7\x9a\xe4\x86\x00\x00\x00\x00IEND\xaeB`\x82",
                        media_type="image/png", headers={"Cache-Control": "max-age=86400"})
    last_err = None
    for url_tmpl in urls:
        url = url_tmpl.format(z=z, x=x, y=y)
        try:
            client = _get_client()
            r = await client.get(url)
            if r.status_code == 200 and len(r.content) > 100:
                ctype = r.headers.get("content-type", "image/png")
                return Response(content=r.content, media_type=ctype,
                                headers={"Cache-Control": "max-age=86400"})
            last_err = f"{r.status_code}"
        except Exception as e:
            last_err = str(e)
            continue
    # transparent 1x1 PNG fallback
    return Response(content=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xfc\xff\xff?\x03\x00\x08\xfc\x02\xfe\xa7\x9a\xe4\x86\x00\x00\x00\x00IEND\xaeB`\x82",
                    media_type="image/png", headers={"Cache-Control": "max-age=300"})

@app.get("/api/health")
def health(): return {"status":"ok","ts":datetime.now(timezone.utc).isoformat()}


@app.get("/api/status")
def status():
    return {
        "system":"online","basin":BASIN["name"],"bbox":BASIN["bbox"],
        "cloudburst_active":STATE.cb_active,"breach_active":STATE.breach_active,
        "cb_started":str(STATE.cb_t0 or ""),
        "server_time":datetime.now(timezone.utc).isoformat(),
    }


class ScenarioIn(BaseModel):
    scenario: str = "cloudburst"
    intensity_multiplier: float = 1.0

@app.post("/api/scenario/trigger")
def trigger(s: ScenarioIn):
    global STATE
    now = datetime.now(timezone.utc).isoformat()
    if s.scenario == "reset":
        STATE.cb_active = False; STATE.breach_active = False; STATE.cb_t0 = None; STATE.cb_age_hr = -1
        bus.publish("events.cloudburst",{"triggered":False,"type":"reset"})
        return {"status":"reset"}
    if s.scenario == "cloudburst":
        STATE.cb_active = True; STATE.cb_t0 = None; STATE.cb_age_hr = -1; STATE.breach_active = False
        bus.publish("events.cloudburst",{"triggered":True,"type":"cloudburst","ts":now})
        add_notification(STATE,"critical","authority","SDMA/DDMA",
            "CLOUDBURST DETECTED near Kedarnath — activating flood response protocol.")
        return {"status":"triggered","scenario":"cloudburst"}
    if s.scenario == "breach":
        STATE.breach_active = True; STATE.cb_active = True
        if STATE.cb_t0 is None: STATE.cb_t0 = None; STATE.cb_age_hr = -1
        bus.publish("events.dam",{"dam_name":"Chorabari Tal","breach":True,"outflow":1500.0,"ts":now})
        add_notification(STATE,"critical","authority","SDMA/DDMA",
            "DAM/Lake BREACH: Chorabari Tal — flash flood wave propagating downstream. Immediate evacuation.")
        return {"status":"triggered","scenario":"breach"}
    return {"error":"unknown scenario"}


@app.get("/api/static/layer/{layer}")
def static_layer(layer: str):
    allowed = {"rivers","roads","settlements","pois"}
    if layer not in allowed: return {"error":"not found"}
    gdf = fallback_osm(layer)
    return json.loads(gdf.to_json())


@app.get("/api/sensors/latest")
def sensors():
    rain=[]; wl=[]
    for g in STATE.gauges:
        code,name,typ,lon,lat,dl,lvl,mmhr = g
        if typ == "rain":
            rain.append({"code":code,"name":name,"lon":lon,"lat":lat,"value_mmhr":mmhr})
        else:
            wl.append({"code":code,"name":name,"lon":lon,"lat":lat,"level_m":lvl,"danger_level":dl})
    return {"rainfall":rain,"water_level":wl}


@app.get("/api/risk/latest")
def risk_latest():
    r = STATE.risk
    if r is None: return {"ts":None,"scores":[]}
    H,W,tr = STATE.hydro["H"],STATE.hydro["W"],STATE.hydro["transform"]
    scores=[]
    step=2
    for rr in range(0,H,step):
        for cc in range(0,W,step):
            s = float(r[rr,cc])
            if s > 15:
                scores.append({"x":cc,"y":rr,"score":round(s,1)})
    return {"ts":datetime.now(timezone.utc).isoformat(),"scores":scores}


@app.get("/api/inundation/latest/{horizon}")
def inundation_latest(horizon: int):
    if horizon not in (1,3,6,12): return {"error":"horizon must be 1/3/6/12"}
    d = STATE.inundation.get(horizon)
    if d is None: return {"horizon_hr":horizon,"depths":[]}
    H,W = d.shape
    depths=[]
    step=1
    for rr in range(0,H,step):
        for cc in range(0,W,step):
            v = float(d[rr,cc])
            if v > 0.1:
                depths.append({"x":cc,"y":rr,"depth":round(v,2)})
    return {"horizon_hr":horizon,"ts":datetime.now(timezone.utc).isoformat(),"depths":depths}


@app.get("/api/impact/latest")
def impact_latest():
    # Build synthetic pop grid (cached)
    if not hasattr(STATE,"_pop_cache"):
        H,W,tr = STATE.hydro["H"],STATE.hydro["W"],STATE.hydro["transform"]
        cols=np.arange(W); rows=np.arange(H)
        CC,RR=np.meshgrid(cols,rows)
        lons = tr.c+(CC+0.5)*tr.a+(RR+0.5)*tr.b
        lats = tr.f+(CC+0.5)*tr.d+(RR+0.5)*tr.e
        pop = np.zeros((H,W),dtype=np.int32)
        for _,srow in STATE.settlements.iterrows():
            spop = int(srow.population or 1000)
            g = np.exp(-((lons-srow.geometry.x)**2+(lats-srow.geometry.y)**2)/(2*0.012**2))
            pop += (g*spop/(g.sum()+1e-9)).astype(np.int32)
        pop = pop + RNG.poisson(0.3,pop.shape)*(STATE.hydro["dem"]<3500)
        pop = np.where(STATE.hydro["dem"]<800,0,pop)
        pop = np.where(STATE.hydro["dem"]>4500,0,pop)
        STATE._pop_cache = pop
    pop = STATE._pop_cache
    depth12 = STATE.inundation.get(12)
    total = int(pop[depth12>0.3].sum()) if depth12 is not None else 0
    at_risk = []
    infra = []
    H,W,tr = STATE.hydro["H"],STATE.hydro["W"],STATE.hydro["transform"]
    if depth12 is not None:
        for _,s in STATE.settlements.iterrows():
            c=int((s.geometry.x-tr.c)/tr.a); r=int((s.geometry.y-tr.f)/tr.e)
            if 0<=c<W and 0<=r<H:
                r0,r1=max(0,r-6),min(H,r+6); c0,c1=max(0,c-6),min(W,c+6)
                if depth12[r0:r1,c0:c1].max()>0.3:
                    at_risk.append({"name":s["name"],"lon":s.geometry.x,"lat":s.geometry.y,
                                    "pop":int(s.population or 0)})
        for _,p in STATE.pois.iterrows():
            c=int((p.geometry.x-tr.c)/tr.a); r=int((p.geometry.y-tr.f)/tr.e)
            if 0<=c<W and 0<=r<H:
                r0,r1=max(0,r-6),min(H,r+6); c0,c1=max(0,c-6),min(W,c+6)
                if depth12[r0:r1,c0:c1].max()>0.3:
                    infra.append({"name":p["name"],"category":p["category"],
                                  "lon":p.geometry.x,"lat":p.geometry.y})
    return {"total_people_estimate":total,"settlements_at_risk":at_risk,
            "crit_infra_at_risk":infra}


@app.get("/api/dispatch/latest")
def dispatch_latest():
    return {"plan":STATE.dispatch_plan}


@app.get("/api/data/resources")
def resources():
    out=[]
    for r in [{"kind":"team","name":"SDRF Rudraprayag","count":6,"lon":78.968,"lat":30.508},
              {"kind":"team","name":"ITBP Phata","count":4,"lon":79.092,"lat":30.672},
              {"kind":"boat","name":"Phata boats","count":6,"lon":79.090,"lat":30.671},
              {"kind":"boat","name":"Rudraprayag boats","count":12,"lon":78.966,"lat":30.506},
              {"kind":"ambulance","name":"Rudraprayag ambulance","count":8,"lon":78.965,"lat":30.505}]:
        out.append(r)
    # Add shelters
    for s in [{"kind":"shelter","name":"Rudraprayag stadium","count":5000,"lon":78.965,"lat":30.505},
              {"kind":"shelter","name":"Guptkashi town hall","count":3000,"lon":79.075,"lat":30.645},
              {"kind":"shelter","name":"Phata shelter","count":2000,"lon":79.090,"lat":30.670}]:
        out.append(s)
    return out


@app.get("/api/alerts/notifications")
def notifications(limit: int = 50):
    return STATE.notifications[:limit]


@app.get("/events")
async def events(request: Request):
    q = bus.subscribe()
    async def gen():
        yield {"event":"hello","data":json.dumps({"ok":True})}
        while True:
            if await request.is_disconnected():
                bus.unsubscribe(q); break
            try:
                ch,payload = q.get(timeout=1.0)
                yield {"event":ch.replace("events.",""),"data":json.dumps(payload)}
            except queue.Empty:
                yield {"event":"ping","data":"{}"}
    return EventSourceResponse(gen())


# ══════════════════════════════════════════════════════════════════
# DASHBOARD (embedded HTML with MapLibre)
# ══════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Flood Response — Mandakini</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://unpkg.com/maplibre-gl@4.1.0/dist/maplibre-gl.css" rel="stylesheet"/>
<script src="https://unpkg.com/maplibre-gl@4.1.0/dist/maplibre-gl.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body,html,#app{width:100%;height:100%;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
.app{display:grid;grid-template-columns:1fr 380px;grid-template-rows:56px 1fr;height:100vh}
.header{grid-column:1/span2;background:linear-gradient(90deg,#0b2545,#13315c);color:#fff;display:flex;align-items:center;padding:0 20px;gap:14px;box-shadow:0 2px 6px rgba(0,0,0,0.2);z-index:10}
.header h1{font-size:17px;font-weight:600}
.header .basin{font-size:12px;color:#8ecae6;margin-left:10px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.dot.ok{background:#4caf50;box-shadow:0 0 8px #4caf50}
.dot.alert{background:#ff5722;box-shadow:0 0 8px #ff5722;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.ctrls{margin-left:auto;display:flex;gap:8px}
.ctrls button{padding:6px 14px;border:none;border-radius:4px;cursor:pointer;font-weight:600;font-size:13px}
.ctrls button:hover{transform:translateY(-1px)}
.b-red{background:#d62828;color:#fff}.b-org{background:#f77f00;color:#fff}.b-gry{background:#6c757d;color:#fff}
.map{position:relative}
.sb{background:#f8f9fa;border-left:1px solid #dee2e6;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px}
.pan{background:#fff;border-radius:8px;padding:10px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}
.pan h3{font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:#495057;margin-bottom:6px;border-bottom:2px solid #e9ecef;padding-bottom:4px}
.mr{display:flex;justify-content:space-between;padding:3px 0;font-size:13px}
.mv{font-weight:700;color:#0b2545}.mv.c{color:#d62828}.mv.w{color:#f77f00}.mv.o{color:#2a9d8f}
.nl{font-size:11px;max-height:200px;overflow-y:auto}
.nt{padding:5px 7px;margin-bottom:3px;border-left:4px solid #457b9d;background:#f1faee;border-radius:2px}
.nt.c{border-color:#d62828;background:#fff0f0}.nt.w{border-color:#f77f00;background:#fff8ec}
.nt small{color:#6c757d;display:block;margin-top:1px}
.zl{font-size:11px;max-height:150px;overflow-y:auto}
.zl .zi{padding:4px;border-bottom:1px solid #f1f3f5;display:flex;justify-content:space-between}
.hs{position:absolute;top:10px;right:10px;background:#fff;padding:8px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.2);z-index:5;font-size:12px}
.hs button{padding:3px 8px;border:1px solid #ced4da;background:#fff;cursor:pointer;border-radius:3px;font-size:11px;margin:1px}
.hs button.on{background:#13315c;color:#fff;border-color:#13315c}
.lt{position:absolute;top:10px;left:10px;background:rgba(255,255,255,0.92);padding:8px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.2);z-index:5;font-size:11px}
.lt label{display:block;padding:1px 0;cursor:pointer}
.lg{position:absolute;bottom:20px;left:10px;background:rgba(255,255,255,0.95);padding:8px;border-radius:6px;font-size:11px;z-index:5;box-shadow:0 2px 6px rgba(0,0,0,0.2)}
.lg .sw{display:inline-block;width:12px;height:12px;margin-right:5px;border-radius:2px;vertical-align:middle}
</style></head>
<body><div id="app"><div class="app">
<div class="header">
<span class="dot ok" id="statusDot"></span>
<h1>Cloudburst → Flash Flood Response</h1><span class="basin">Mandakini River · Kedarnath, Uttarakhand</span>
<div class="ctrls">
<button class="b-red" onclick="trig('cloudburst')">⚡ Trigger Cloudburst</button>
<button class="b-org" onclick="trig('breach')">🌊 Dam Breach</button>
<button class="b-gry" onclick="trig('reset')">Reset</button>
</div></div>
<div class="map" id="map"></div>
<div class="sb" id="sb">
<div class="pan"><h3>System</h3>
<div class="mr"><span>State</span><span class="mv o" id="sState">● Monitoring</span></div>
<div class="mr"><span>Peak rain</span><span class="mv o" id="sRain">0 mm/hr</span></div>
<div class="mr"><span>Peak river level</span><span class="mv o" id="sWL">0 m</span></div>
</div>
<div class="pan"><h3>Flood Risk</h3>
<div class="mr"><span>Max score</span><span class="mv o" id="sRisk">—</span></div>
<div class="mr"><span>High-risk cells</span><span class="mv" id="sHigh">0</span></div>
<div class="zl" id="sTopZones"></div></div>
<div class="pan"><h3>People &amp; Infrastructure</h3>
<div class="mr"><span>People at risk</span><span class="mv o" id="sPop">0</span></div>
<div class="mr"><span>Settlements</span><span class="mv" id="sSett">0</span></div>
<div class="mr"><span>Critical facilities</span><span class="mv" id="sPOI">0</span></div>
<div class="zl" id="sSettList"></div></div>
<div class="pan"><h3>Optimized Dispatch</h3>
<div class="mr"><span>Avg response</span><span class="mv o" id="sAvgTT">—</span></div>
<div class="mr"><span>vs baseline</span><span class="mv o" id="sBase">—</span></div>
<div class="mr"><span>Shortfalls</span><span class="mv o" id="sShort">—</span></div>
<div class="zl" id="sAssn"></div></div>
<div class="pan"><h3>Notifications</h3>
<div class="nl" id="sNotif"></div></div>
</div></div></div>
<script>
const BBOX=[78.93,30.49,79.20,30.77], CENTER=[79.055,30.630];
const RES_DEG=10/3600;
function p2ll(x,y){
 const w=BBOX[0],s=BBOX[1],e=BBOX[2],n=BBOX[3];
 const W=Math.round((e-w)/RES_DEG),H=Math.round((n-s)/RES_DEG);
 return[w+(x/W)*(e-w),n-(y/H)*(n-s)]
}
function ll2p(lon,lat){
 const w=BBOX[0],s=BBOX[1],e=BBOX[2],n=BBOX[3];
 const W=Math.round((e-w)/RES_DEG),H=Math.round((n-s)/RES_DEG);
 return[Math.max(0,Math.min(W-1,Math.floor((lon-w)/(e-w)*W))),
        Math.max(0,Math.min(H-1,Math.floor((n-lat)/(n-s)*H)))]
}

// Basemap styles — all proxied through /tiles/* to avoid browser 403
// (OSM/CARTO/Esri block requests with default browser UA in some sandboxes).
// NO API KEY needed for any of these providers.
const BASEMAPS = {
  topo: {
    label: 'OpenTopoMap (terrain)',
    tiles: ['/tiles/topo/{z}/{x}/{y}.png'],
    attr: '© OpenStreetMap contributors, © OpenTopoMap (CC-BY-SA)'
  },
  esri_topo: {
    label: 'Esri Topographic',
    tiles: ['/tiles/esri_topo/{z}/{x}/{y}.png'],
    attr: 'Sources: Esri, HERE, Garmin, © OSM'
  },
  esri_sat: {
    label: 'Esri Satellite',
    tiles: ['/tiles/esri_img/{z}/{x}/{y}.png'],
    attr: 'Sources: Esri, Maxar, Earthstar Geographics'
  },
};
let currentBasemap = 'topo';

function buildStyle(bmKey) {
  const bm = BASEMAPS[bmKey];
  return {
    version: 8,
    sources: {
      basemap: { type:'raster', tiles: bm.tiles, tileSize:256, attribution: bm.attr }
    },
    layers: [{ id:'basemap', type:'raster', source:'basemap' }]
  };
}

const map=new maplibregl.Map({
  container:'map',
  style: buildStyle(currentBasemap),
  // Frames the Kedarnath → Rudraprayag corridor perfectly
  center:[79.055, 30.630],
  zoom:10.2,
  minZoom:9, maxZoom:15,
  maxBounds: [[78.6,30.3],[79.6,31.1]]
});
map.addControl(new maplibregl.NavigationControl({showCompass:false}),'bottom-right');
map.addControl(new maplibregl.ScaleControl({unit:'metric'}),'bottom-left');

let horizon=3, layers={rivers:true,roads:true,settlements:true,risk:true,inundation:true,resources:true,sensors:true};
let state={rainfall:[],water_level:[],risk:null,inundation:null,impact:null,dispatch:null,notifs:[],cb:false};

map.on('load',async()=>{
 await addStatic();
 addHSControls();
 addLegend();
 refresh();
 setInterval(refresh,3000);
 const es=new EventSource('/events');
 es.addEventListener('risk',()=>refresh());
 es.addEventListener('inundation',()=>{fetchInundation();renderInundation();renderImpact()});
 es.addEventListener('dispatch',()=>{fetchDispatch();renderDispatch()});
 es.addEventListener('alert',e=>{try{const d=JSON.parse(e.data);state.notifs.unshift(d);state.notifs=state.notifs.slice(0,30);renderNotifs()}catch(err){}});
 es.addEventListener('cloudburst',()=>refresh());
 es.addEventListener('rain',e=>{});
 es.addEventListener('water',e=>{});
});

async function addStatic(){
 for(const lyr of ['rivers','roads','settlements','pois']){
   const fc=await fetch('/api/static/layer/'+lyr).then(r=>r.json());
   map.addSource(lyr,{type:'geojson',data:fc});
   if(lyr==='rivers') {
     // outer glow/casing so river is readable on satellite too
     map.addLayer({id:lyr+'-glow',type:'line',source:lyr,paint:{
       'line-color':'#0b5fff','line-width':['interpolate',['linear'],['zoom'],9,8,12,14,15,22],
       'line-opacity':0.25,'line-blur':3
     }});
     map.addLayer({id:lyr+'-l',type:'line',source:lyr,paint:{
       'line-color':'#1f7ae0','line-width':['interpolate',['linear'],['zoom'],9,3,12,5,15,9],
       'line-opacity':0.95
     }});
     map.addLayer({id:lyr+'-lab',type:'symbol',source:lyr,layout:{
       'text-field':['get','name'],'text-size':11,'text-anchor':'center',
       'symbol-placement':'line','text-letter-spacing':0.15,'text-rotate':0,
       'text-allow-overlap':false,'text-ignore-placement':false
     },paint:{
       'text-color':'#073b8a','text-halo-color':'#ffffff','text-halo-width':1.5,'text-halo-blur':0.5
     }});
   }
   if(lyr==='roads') map.addLayer({id:lyr+'-l',type:'line',source:lyr,paint:{'line-color':'#a33','line-width':1.5,'line-dasharray':[2,1],'line-opacity':0.6}});
   if(lyr==='settlements'){
     map.addLayer({id:lyr+'-l',type:'circle',source:lyr,paint:{'circle-radius':4,'circle-color':'#333','circle-stroke-color':'#fff','circle-stroke-width':1}});
     map.addLayer({id:lyr+'-lab',type:'symbol',source:lyr,layout:{'text-field':['get','name'],'text-size':10,'text-offset':[0,0.8],'text-anchor':'top'},paint:{'text-color':'#222','text-halo-color':'#fff','text-halo-width':1}});
   }
   if(lyr==='pois') map.addLayer({id:lyr+'-l',type:'circle',source:lyr,paint:{'circle-radius':['match',['get','category'],'hospital',6,'school',4,'police',5,'fire',5,4],'circle-color':['match',['get','category'],'hospital','#e63946','school','#457b9d','police','#2a9d8f','fire','#f77f00','#888'],'circle-stroke-width':1.5,'circle-stroke-color':'#fff'}});
 }
 addResources();
}

function addResources(){
 const feats=[];
 [
  ['SDRF Rudraprayag',78.968,30.508,'team'],['ITBP Phata',79.092,30.672,'team'],
  ['Phata boats',79.090,30.671,'boat'],['Rudraprayag boats',78.966,30.506,'boat'],
  ['Rudraprayag ambulance',78.965,30.505,'amb'],
  ['Rudraprayag stadium shelter',78.965,30.505,'shelter'],
  ['Guptkashi shelter',79.075,30.645,'shelter'],['Phata shelter',79.090,30.670,'shelter'],
 ].forEach(([n,lon,lat,k])=>feats.push({type:'Feature',properties:{kind:k,name:n},geometry:{type:'Point',coordinates:[lon,lat]}}));
 map.addSource('res',{type:'geojson',data:{type:'FeatureCollection',features:feats}});
 map.addLayer({id:'res-l',type:'circle',source:'res',paint:{'circle-radius':7,'circle-color':['match',['get','kind'],'shelter','#1a9850','team','#2a9d8f','boat','#457b9d','#f77f00'],'circle-stroke-width':2,'circle-stroke-color':'#fff'}});
}

function addHSControls(){
 // Horizon selector
 const d=document.createElement('div');d.className='hs';
 d.innerHTML='<label><b>Flood prediction horizon</b></label><div class="hb">'+[1,3,6,12].map(h=>`<button onclick="setH(${h})" id="b${h}" class="${h===horizon?'on':''}">${h}h</button>`).join('')+'</div>';
 document.getElementById('map').appendChild(d);
 // Basemap switcher
 const bm=document.createElement('div');bm.className='hs';
 bm.style.top='auto';bm.style.bottom='10px';bm.style.right='10px';
 let bmHtml='<label><b>Basemap</b></label><div style="display:flex;flex-direction:column;gap:2px;margin-top:4px">';
 Object.keys(BASEMAPS).forEach(k=>{
  bmHtml+=`<button id="bm-${k}" onclick="setBasemap('${k}')" style="text-align:left;padding:3px 8px;border:1px solid ${k===currentBasemap?'#13315c':'#ced4da'};background:${k===currentBasemap?'#13315c':'#fff'};color:${k===currentBasemap?'#fff':'#222'};cursor:pointer;border-radius:3px;font-size:11px">${BASEMAPS[k].label}</button>`;
 });
 bmHtml+='</div>';
 bm.innerHTML=bmHtml;
 document.getElementById('map').appendChild(bm);
 // Layer toggles
 const l=document.createElement('div');l.className='lt';
 l.innerHTML='<b style="display:block;margin-bottom:3px;font-size:12px">Layers</b>'+
  ['rivers','roads','settlements','risk','inundation','resources'].map(k=>`<label><input type="checkbox" checked onchange="toggleLayer('${k}',this.checked)"> ${k}</label>`).join('');
 document.getElementById('map').appendChild(l);
 const lg=document.createElement('div');lg.className='lg';
 lg.style.bottom='auto';lg.style.top='80px';
 lg.innerHTML='<b style="display:block;margin-bottom:3px">Legend</b>'+
  '<div><span class="sw" style="background:rgba(255,0,0,0.6)"></span>Extreme risk</div>'+
  '<div><span class="sw" style="background:rgba(255,165,0,0.6)"></span>High risk</div>'+
  '<div><span class="sw" style="background:rgba(0,100,255,0.5)"></span>Flood depth</div>'+
  '<div><span class="sw" style="background:#1a9850"></span>Shelters/Teams</div>';
 document.getElementById('map').appendChild(lg);
}
window.setH=h=>{horizon=h;[1,3,6,12].forEach(x=>document.getElementById('b'+x).className=x===h?'on':'');renderInundation()};
window.toggleLayer=(k,v)=>{
  layers[k]=v;
  // static base layers have MapLibre layer ids <k>-l; rivers also has glow + label
  ['-l','-glow','-lab'].forEach(suf=>{
    const lid=k+suf;
    if(map.getLayer(lid)) map.setLayoutProperty(lid,'visibility',v?'visible':'none');
  });
  renderRisk();renderInundation();
};
window.setBasemap=k=>{
 currentBasemap=k;
 map.setStyle(buildStyle(k));
 map.once('styledata',()=>{
   // Re-add sources & layers after style changes
   setTimeout(async()=>{await addStatic();renderRisk();renderInundation();renderDispatch&&renderDispatch();},300);
 });
 Object.keys(BASEMAPS).forEach(bk=>{
   const el=document.getElementById('bm-'+bk);
   if(el){el.style.background=bk===k?'#13315c':'#fff';el.style.color=bk===k?'#fff':'#222';el.style.borderColor=bk===k?'#13315c':'#ced4da';}
 });
};

async function refresh(){
 try{
  const s=await fetch('/api/status').then(r=>r.json());state.cb=s.cloudburst_active;
  document.getElementById('statusDot').className='dot '+(state.cb?'alert':'ok');
  document.getElementById('sState').textContent=state.cb?'⚠ CLOUDBURST ACTIVE':s.breach_active?'⚠ BREACH ACTIVE':'● Monitoring';
  document.getElementById('sState').className='mv '+(state.cb||s.breach_active?'c':'o');
  const sn=await fetch('/api/sensors/latest').then(r=>r.json());state.rainfall=sn.rainfall;state.water_level=sn.water_level;
  const mxr=Math.max(0,...sn.rainfall.map(r=>r.value_mmhr||0));const mxw=Math.max(0,...sn.water_level.map(w=>w.level_m||0));
  document.getElementById('sRain').textContent=mxr.toFixed(1)+' mm/hr';document.getElementById('sRain').className='mv '+(mxr>100?'c':mxr>50?'w':'o');
  document.getElementById('sWL').textContent=mxw.toFixed(2)+' m';document.getElementById('sWL').className='mv '+(mxw>4?'c':mxw>2.5?'w':'o');
 }catch(e){}
 await Promise.all([fetchRisk(),fetchInundation()]);
 renderRisk();renderInundation();
 await Promise.all([fetchImpact(),fetchDispatch()]);
 renderImpact();renderDispatch();
 fetchNotifs();renderNotifs();
}

async function fetchRisk(){state.risk=await fetch('/api/risk/latest').then(r=>r.json())}
async function fetchInundation(){state.inundation=await fetch('/api/inundation/latest/'+horizon).then(r=>r.json())}
async function fetchImpact(){state.impact=await fetch('/api/impact/latest').then(r=>r.json())}
async function fetchDispatch(){state.dispatch=await fetch('/api/dispatch/latest').then(r=>r.json())}
async function fetchNotifs(){state.notifs=await fetch('/api/alerts/notifications').then(r=>r.json())}

function renderRisk(){
 const r=state.risk;if(!r)return;
 const scores=r.scores||[];
 const maxS=scores.reduce((m,s)=>Math.max(m,s.score),0);
 document.getElementById('sRisk').textContent=maxS.toFixed(0)+'/100';
 document.getElementById('sRisk').className='mv '+(maxS>80?'c':maxS>50?'w':'o');
 document.getElementById('sHigh').textContent=scores.filter(s=>s.score>70).length;
 document.getElementById('sTopZones').innerHTML=scores.filter(s=>s.score>40).sort((a,b)=>b.score-a.score).slice(0,6).map(s=>{
  const sc=s.score>80?'c':s.score>60?'w':'o';
  return `<div class="zi"><span>Cell (${s.x},${s.y})</span><span class="mv ${sc}">${s.score.toFixed(0)}</span></div>`;
 }).join('');
 // Source
 const fc={type:'FeatureCollection',features:scores.filter(s=>s.score>20).map(s=>{const ll=p2ll(s.x,s.y);return{type:'Feature',properties:{score:s.score},geometry:{type:'Point',coordinates:ll}}})};
 if(map.getSource('risk'))map.getSource('risk').setData(fc);
 else map.addSource('risk',{type:'geojson',data:fc}),map.addLayer({id:'risk-l',type:'circle',source:'risk',paint:{'circle-radius':['interpolate',['linear'],['get','score'],20,3,60,8,100,14],'circle-color':['interpolate',['linear'],['get','score'],20,'rgba(255,255,0,0.3)',50,'rgba(255,165,0,0.5)',70,'rgba(255,80,0,0.6)',90,'rgba(220,0,0,0.75)'],'circle-blur':0.5}},'settlements-l');
 if(map.getLayer('risk-l'))map.setLayoutProperty('risk-l','visibility',layers.risk?'visible':'none');
}

function renderInundation(){
 const d=state.inundation;if(!d)return;
 const depths=d.depths||[];
 const fc={type:'FeatureCollection',features:depths.map(d=>{const ll=p2ll(d.x,d.y);return{type:'Feature',properties:{depth:d.depth},geometry:{type:'Point',coordinates:ll}}})};
 if(map.getSource('inun'))map.getSource('inun').setData(fc);
 else map.addSource('inun',{type:'geojson',data:fc}),map.addLayer({id:'inun-l',type:'circle',source:'inun',paint:{'circle-radius':['interpolate',['linear'],['get','depth'],0.1,4,1,8,3,12,8,16],'circle-color':['interpolate',['linear'],['get','depth'],0.2,'rgba(200,220,255,0.4)',0.5,'rgba(100,150,255,0.6)',2,'rgba(0,80,200,0.75)',5,'rgba(0,0,120,0.85)'],'circle-blur':0.3}},'risk-l');
 if(map.getLayer('inun-l'))map.setLayoutProperty('inun-l','visibility',layers.inundation?'visible':'none');
 // Draw dispatch lines from resources to zones
 if(state.dispatch && state.dispatch.plan){
  const lines=[];
  state.dispatch.plan.assignments.forEach(a=>lines.push({type:'Feature',geometry:{type:'LineString',coordinates:[[a.resource_lon,a.resource_lat],[a.zone_lon,a.zone_lat]]}}));
  if(map.getSource('dl'))map.getSource('dl').setData({type:'FeatureCollection',features:lines});
  else map.addSource('dl',{type:'geojson',data:{type:'FeatureCollection',features:lines}}),map.addLayer({id:'dl-l',type:'line',source:'dl',paint:{'line-color':'#1a9850','line-width':2,'line-opacity':0.7,'line-dasharray':[2,1]}});
 }
 if(map.getLayer('dl-l'))map.setLayoutProperty('dl-l','visibility',layers.resources?'visible':'none');
}

function renderImpact(){
 const i=state.impact;if(!i)return;
 document.getElementById('sPop').textContent=(i.total_people_estimate||0).toLocaleString();
 document.getElementById('sPop').className='mv '+(i.total_people_estimate>5000?'c':i.total_people_estimate>1000?'w':'o');
 document.getElementById('sSett').textContent=(i.settlements_at_risk||[]).length;
 document.getElementById('sPOI').textContent=(i.crit_infra_at_risk||[]).length;
 document.getElementById('sSettList').innerHTML=(i.settlements_at_risk||[]).slice(0,8).map(s=>`<div class="zi"><span>${s.name}</span><span>~${s.pop?.toLocaleString?.()||'?'}</span></div>`).join('');
}

function renderDispatch(){
 const p=state.dispatch?.plan;
 if(!p){return}
 document.getElementById('sAvgTT').textContent=(p.avg_travel_minutes||0).toFixed(0)+' min';
 document.getElementById('sBase').textContent=(p.baseline_avg_minutes||0).toFixed(0)+' min';
 const imp=((p.baseline_avg_minutes||0)-(p.avg_travel_minutes||0))/(p.baseline_avg_minutes||1)*100;
 document.getElementById('sBase').textContent+=' ('+(imp>0?'-':'')+imp.toFixed(0)+'%)';
 document.getElementById('sShort').textContent=p.shortfalls?.length||0;
 document.getElementById('sShort').className='mv '+(p.shortfalls?.length>0?'c':'o');
 document.getElementById('sAssn').innerHTML=(p.assignments||[]).slice(0,8).map(a=>`<div class="zi"><span>${a.units}× ${a.resource_kind} → ${a.zone_name}</span><span>${a.travel_minutes.toFixed(0)}m</span></div>`).join('')+
  (p.shortfalls||[]).slice(0,3).map(s=>`<div class="zi" style="color:#d62828"><span>⚠ ${s.zone_name}</span><span>+${s.teams_short}t/${s.boats_short}b</span></div>`).join('');
}

function renderNotifs(){
 const n=state.notifs||[];
 document.getElementById('sNotif').innerHTML=n.slice(0,15).map(n=>`<div class="nt ${n.severity==='critical'?'c':n.severity==='warn'?'w':''}">[${n.channel}] ${n.message}<small>${new Date(n.ts).toLocaleTimeString()}</small></div>`).join('');
}

async function trig(sc){
 await fetch('/api/scenario/trigger',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scenario:sc})});
 setTimeout(refresh,1000);setTimeout(refresh,5000);setTimeout(refresh,10000);setTimeout(refresh,18000);
}
</script>
</body></html>"""


@app.get("/")
def root():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/docs")
def docs_redir():
    return HTMLResponse("""<html><body><h1>Flood Response API</h1>
<p>See <a href="/health">/health</a> · <a href="/api/status">/api/status</a> · <a href="/api/sensors/latest">sensors</a> · <a href="/api/risk/latest">risk</a></p>
<p>Interactive dashboard: <a href="/">/</a></p></body></html>""")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    global STATE
    print("═"*70)
    print("  AI Cloudburst → Flash Flood Response System")
    print("  Basin: Mandakini River (Kedarnath, Uttarakhand, India)")
    print("═"*70)
    print("[setup] Generating terrain...")
    dem, transform = make_synthetic_dem()
    print(f"[setup] DEM: {dem.shape[1]}×{dem.shape[0]}, elev {dem.min():.0f}–{dem.max():.0f} m")
    settlements = fallback_osm("settlements")
    pois = fallback_osm("pois")
    print(f"[setup] {len(settlements)} settlements, {len(pois)} POIs, fallback OSM loaded.")

    print("[setup] Computing hydrology layers...")
    hydro = compute_hydrology(dem, transform, settlements)
    print(f"[setup] Slope {hydro['slope'].min():.1f}–{hydro['slope'].max():.1f}°, "
          f"flow acc max {hydro['flowacc'].max():.0f}")

    # Resources & shelters (hardcoded plausible locations)
    STATE = State(hydro, settlements, pois, [], [])

    stop_event = threading.Event()
    sim_t = threading.Thread(target=simulator_loop, args=(STATE, stop_event), daemon=True)
    work_t = threading.Thread(target=worker_loop, args=(STATE, stop_event), daemon=True)
    sim_t.start(); work_t.start()

    print("\n" + "═"*70)
    print("  ✅ SYSTEM ONLINE")
    print("     Dashboard: http://localhost:8000")
    print("     Click '⚡ Trigger Cloudburst' to start the demo.")
    print("═"*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
