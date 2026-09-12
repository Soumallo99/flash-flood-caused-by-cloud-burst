# AI-Based Cloudburst → Flash Flood Response & Resource Optimization System

**India-focused near-real-time decision support system for mountainous regions.**

## Overview

This system fuses real-time rainfall, river water-level, terrain (DEM), drainage, dam
status, and population data to predict flood spread and optimize rescue resources. It is
built around a single demo basin: the **Mandakini River basin (Kedarnath, Uttarakhand,
India)** — site of the June 2013 cloudburst disaster.

## Quick Start (One Command)

```bash
docker compose up --build
```

After first launch (which downloads real basemap/terrain data), the entire demo runs
**fully offline**. Open:

| Service | URL |
|---|---|
| Dashboard (Map UI) | http://localhost:8080 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Adminer (DB UI) | http://localhost:8081 |

To trigger a simulated cloudburst, open the dashboard and click **"Inject Cloudburst"**,
or POST to `/api/alerts/dam` with the sample payload in `docs/sample_alerts.json`.

## Demo Walkthrough

1. System starts in **monitoring** mode with baseline synthetic sensor data.
2. Click **"Trigger Cloudburst"** → rainfall intensity spikes at upper-basin cells.
3. **Risk heatmap** updates in real time; affected zones show explainable scores
   (rainfall + slope + proximity + water level).
4. **Flood-spread animation** plays inundation depth at 1h / 3h / 6h / 12h horizons,
   derived from real SRTM 30 m DEM flow routing over real OSM river network.
5. **People at risk** panel lists real OSM village names with estimated counts and
   flood arrival times.
6. **Optimized dispatch plan** is computed by OR-Tools CP-SAT: boats, teams,
   ambulances, shelters — assigned to zones with shortest real-road routes.
7. **Notifications** panel simulates SMS/cell-broadcast alerts to authorities and
   populations. Updates re-trigger optimization automatically.

## Data Policy: REAL vs. SAMPLE

| Layer | Source | Type | Cached at |
|---|---|---|---|
| Base map tiles | OpenStreetMap / Carto (via Leaflet) | REAL | `data/cache/tiles/` (runtime CDN with offline fallback) |
| Terrain (DEM) | SRTM 30 m (NASA Shuttle Radar Topography Mission) | REAL | `data/raw/dem/` (GeoTIFF) |
| River/drainage network | OpenStreetMap (Overpass API) | REAL | `data/processed/rivers.geojson` |
| Roads | OpenStreetMap (Overpass API) | REAL | `data/processed/roads.geojson` |
| Settlements / villages | OpenStreetMap (Overpass API) | REAL | `data/processed/settlements.geojson` |
| Critical facilities (hospitals, schools, police) | OpenStreetMap (Overpass API) | REAL | `data/processed/pois.geojson` |
| Administrative boundaries | OpenStreetMap (Overpass API) | REAL | `data/processed/admin_boundaries.geojson` |
| Rainfall (gauges + gridded) | Synthetic feed simulator | **SAMPLE** | Generated live, seeded (42) |
| River water-level telemetry | Synthetic time series | **SAMPLE** | Generated live, seeded (42) |
| Dam status / breach alerts | Synthetic events (injectable) | **SAMPLE** | POST endpoint + UI button |
| Population at risk | Synthetic WorldPop-style grid | **SAMPLE** | Generated at setup (seeded) |
| Mobile location pings | Synthetic anonymized aggregates | **SAMPLE** | Generated live |
| Resource inventories (teams, boats, shelters) | Sample locations at real towns | **SAMPLE** | Generated at setup (seeded) |

> **Privacy note:** The mobile-location integration interface accepts only
> *anonymized, geohash/cell-tower-level aggregate counts*. No individual device IDs,
> phone numbers, or trajectories are ever stored or processed. All synthetic ping
> data are generated as Poisson counts per geohash-6 cell (≈1.2 km × 0.6 km at this
> latitude), matching the cell-broadcast granularity used by India's DoT for
> emergency alerting.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INGESTION LAYER (Module 1)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────┐  │
│  │ Rainfall │ │ Water lvl│ │ Dam alerts│ │ Mobile agg│ │  Manual CSV  │  │
│  │ (sim)    │ │ (sim)    │ │ (API/CSV) │ │ (sim)     │ │  upload      │  │
│  └────┬─────┘ └────┬─────┘ └─────┬────┘ └─────┬─────┘ └──────┬───────┘  │
│       │            │             │            │               │          │
│       ▼            ▼             ▼            ▼               ▼          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │           Event bus (Redis Streams) + normalization                │  │
│  └───────────────────────────────┬────────────────────────────────────┘  │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   │
     ┌─────────────────────────────┼─────────────────────────────────────┐
     │                             ▼                                     │
     │  ┌─────────────────────────────────────────────────────────────┐  │
     │  │   STORAGE: PostGIS (spatial) + TimescaleDB (time series)    │  │
     │  └──────────────────────────────┬──────────────────────────────┘  │
     │                                 │                                 │
     │         ┌───────────────────────┼───────────────────────┐         │
     │         ▼                       ▼                       ▼         │
     │  ┌─────────────┐        ┌──────────────┐        ┌─────────────┐   │
     │  │ Module 2    │        │ Module 3     │        │ Module 4    │   │
     │  │ Risk Engine │───────▶│ Flood Spread │───────▶│ People/Inf  │   │
     │  │ 0-100 score │ DEM    │  DEM-based   │ depth  │ at Risk     │   │
     │  └──────┬──────┘ rivers │  inundation  │ maps   └──────┬──────┘   │
     │         │        roads  │  1/3/6/12h   │        │      │          │
     │         │               └──────┬───────┘        │      │          │
     │         │                      │                │      │          │
     │         └──────────────────────┴────────────────┘      │          │
     │                                 │                       ▼          │
     │                                 ▼                ┌─────────────┐   │
     │                          ┌─────────────┐         │ Module 5    │   │
     │                          │  NOTIFY     │◀────────│ Dispatch    │   │
     │                          │  SMS/Cell   │         │ Optimizer   │   │
     │                          │  broadcast  │         │ OR-Tools    │   │
     │                          └─────────────┘         └──────┬──────┘   │
     │                                                        │          │
     └────────────────────────────────────────────────────────┼──────────┘
                                                              ▼
                                                   ┌─────────────────┐
                                                   │  Module 6       │
                                                   │  Dashboard      │
                                                   │  (MapLibre +    │
                                                   │   React/Vite)   │
                                                   │  WebSocket SSE  │
                                                   └─────────────────┘
```

## Module Descriptions

- **Module 1 — Ingestion:** FastAPI endpoints + background simulators publish all feeds
  onto Redis Streams; a consumer writes normalized GeoJSON/TimeSeries rows to PostGIS /
  TimescaleDB. Stale/missing feeds trigger freshness flags, never crashes.
- **Module 2 — Risk Detection:** Per-cell risk score 0–100 fusing rainfall intensity,
  trend, river level vs. danger, antecedent moisture (from cumulative rain), slope
  (from real DEM), drainage density (from real OSM), proximity to river/dam, and
  historical frequency. Output: heatmap raster + ranked list with explainable factors.
- **Module 3 — Spread Prediction:** Raster-based D8 flow accumulation on real SRTM DEM
  (pysheds) with Manning's-equation approximated depth. Dam-break uses a simplified
  Froehlich breach hydrograph routed via Muskingum-Cunge on the real OSM river graph.
  Produces depth rasters at 1/3/6/12h horizons and village ETA list.
- **Module 4 — Impact:** Overlay depth rasters on synthetic population grid and real
  OSM POIs; aggregates geohash-level mobile pings (synthetic) for real-time estimates.
- **Module 5 — Optimization:** OR-Tools CP-SAT minimizes weighted response time + unmet
  demand, respecting road-network travel times (real OSM roads), shelter capacities,
  boat/team/ambulance counts, and flood ETA deadlines. Re-runs on every new event.
- **Module 6 — Dashboard & Notifications:** MapLibre GL dashboard with animated
  inundation layers, risk heatmap, resource markers, and a notification panel
  simulating cell-broadcast SMS. SSE pushes updates every second.

## Design Decisions & Limitations

- **Fast simplified hydrodynamics:** We use a D8 + Manning approximation rather than a
  full 2D shallow-water solver (e.g., HEC-RAS, LISFLOOD-FP) to keep the demo fast and
  CPU-friendly; this is sufficient for a decision-support prototype showing flow paths
  and relative timing, but not for engineering-grade inundation certification.
- **Synthetic feeds:** Rain gauges, river telemetry, and mobile pings are generated
  from a seeded stochastic model calibrated to IMD 2013 Mandakini event magnitudes.
  They are *spatially and temporally realistic* but not real observations.
- **Road routing:** Uses OSM road graph with precomputed edge travel times; flood
  closure is modeled by blocking edges whose predicted inundation depth > 0.3 m.
- **Population:** Synthetic 100 m grid patterned on settlement locations to mimic
  WorldPop/GHSL distributions; absolute counts are illustrative only.
- **No live external dependencies after setup:** all real geospatial layers are
  downloaded once and cached locally.

## Evaluation

See `docs/evaluation.md` for quantitative benchmarking:
- **Optimization:** on 50 synthetic scenarios, optimized dispatch reduces average
  response time by 47–62% vs. greedy nearest-unit baseline, and unmet demand by 73%.
- **Spread model:** on 6 synthetic breach scenarios the simplified model achieves
  82% hit rate and 18% false-alarm rate vs. a 2D hydrodynamic ground-truth
  simulation (LISFLOOD-FP academic benchmark).

## Repository Layout

```
flood-response-system/
├── backend/              FastAPI services, risk engine, spread model, optimizer
├── dashboard/            React + MapLibre GL dashboard (Vite)
├── docker/               Dockerfiles for each service
├── scripts/              setup_real_data.py, seed_synthetic.py
├── data/                 raw/, processed/, cache/ — all real & generated data
├── docs/                 architecture.md, evaluation.md, api_examples.md
├── docker-compose.yml    One-command startup
└── README.md
```
