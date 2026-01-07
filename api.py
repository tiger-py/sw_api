from __future__ import annotations

from typing import Dict, List, Literal, Tuple
import re
import time
import zipfile
import csv
import json
import io
import math
import datetime as dt

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sim_core import (
    get_meteo_timeseries,
    simulate_annual_output,
    water_metrics_from_shading_samples,
    get_tilted_timeseries,
    solar_position_approx_utc,
)


# ======================= App =======================

app = FastAPI(title="Solar Output API", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================= Models: Simulation =======================

class SimRequest(BaseModel):
    lat: float = Field(..., description="Latitude in degrees")
    lon: float = Field(..., description="Longitude in degrees")

    # "roof"   => single tilted plane
    # "waves"  => Solar Waves: split into two fixed azimuths (E/W)
    array_type: Literal["roof", "waves"]

    # Roof controls
    azimuth_deg: float = Field(180.0, description="Roof azimuth (deg, ignored for waves)")
    tilt_deg: float = Field(15.0, description="Roof tilt (deg, ignored for waves)")

    # Count + panel geometry
    total_panels: int = Field(..., ge=1)
    panel_width_m: float = Field(..., gt=0)
    panel_height_m: float = Field(..., gt=0)

    # Electrical/thermal params (defaults; can be driven from panels.json later)
    eff_stc: float = Field(0.20, ge=0, le=1)
    noct: float = Field(45.0)
    temp_coeff: float = Field(-0.0038)
    cooling_offset: float = Field(0.0)


class SimResponse(BaseModel):
    annual_energy_kwh: float
    monthly_energy_kwh: Dict[str, float]
    mean_poa_w_m2: float


class ShadingSample(BaseModel):
    time_utc: str
    # Optional but recommended for audit/repro: sun azimuth/elevation used for this tau sample
    az_deg: float | None = None
    el_deg: float | None = None
    tau: float = 1.0
    weight_hours: float

class SimShadedRequest(SimRequest):
    year: int | None = None
    # Optional: width of the modeled water surface (m). Included for auditability; shading samples already encode partial coverage.
    water_width_m: float | None = Field(None, gt=0.0)
    # Sky View Factor for diffuse shading (0..1). If omitted, defaults to 1 (no diffuse blockage).
    svf: float | None = Field(None, ge=0.0, le=1.0)
    svf_n_rays: int | None = Field(None, ge=1)
    svf_scheme: str | None = None
    shading_samples: List[ShadingSample]

class SimShadedResponse(SimResponse):
    water_baseline_kwh_m2: float
    water_shaded_kwh_m2: float
    water_reduction_pct: float


class OrientationSnapshot(BaseModel):
    orientation_key: str
    azimuth_deg: float
    tilt_deg: float
    panel_count: int
    daily_mean_poa_w_m2: float
    daily_sum_poa_w_m2: float
    hourly_poa_w_m2: List[float]


class SnapshotResponse(BaseModel):
    snapshot_day: str
    snapshot_hour_utc: str
    snapshot_sun_az_deg: float
    snapshot_sun_el_deg: float
    orientations: List[OrientationSnapshot]


# ======================= Models: Geocoding =======================

class GeocodeRequest(BaseModel):
    query: str = Field(..., description="Free-text location, e.g. 'Berlin, Germany'")


class GeocodeResult(BaseModel):
    display_name: str
    lat: float
    lon: float


class GeocodeResponse(BaseModel):
    provider: Literal["offline", "nominatim", "open-meteo"]
    results: List[GeocodeResult]


# ======================= Health =======================

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ======================= Helpers =======================

def _normalize_ui_azimuth(deg: float) -> float:
    d = float(deg) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def _orientation_key(az_deg: float, tilt_deg: float) -> str:
    az = ((float(az_deg) % 360.0) + 360.0) % 360.0
    az = round(az, 1)
    tilt = round(float(tilt_deg), 1)
    return f"{az:.1f}|{tilt:.1f}"


def _build_orientations(
    array_type: str,
    azimuth_deg: float,
    tilt_deg: float,
    total_panels: int,
) -> List[Dict[str, float]]:
    orientations: List[Dict[str, float]] = []
    total = max(1, int(total_panels))

    if array_type == "waves":
        # Solar Waves: treat UI azimuth as the facing direction (module normal) of one pitch.
        # The opposite pitch faces 180° away. This keeps azimuth semantics consistent with "roof".
        base = float(azimuth_deg)
        azimuths = [base, base + 180.0]
        counts = [total // 2, total - (total // 2)]
        tilts = [12.5, 12.5]
    else:
        azimuths = [float(azimuth_deg)]
        counts = [total]
        tilts = [float(tilt_deg)]

    for idx, (az, count, tilt_val) in enumerate(zip(azimuths, counts, tilts)):
        if count <= 0:
            continue
        orientations.append(
            {
                "id": f"o{idx}",
                "panel_count": int(count),
                "azimuth_deg": float(az),
                "azimuth_api_deg": _normalize_ui_azimuth(float(az)),
                "tilt_deg": float(tilt_val),
                "orientation_key": _orientation_key(float(az), float(tilt_val)),
            }
        )
    return orientations


# ======================= Simulation =======================

@app.post("/simulate", response_model=SimResponse)
def simulate(req: SimRequest) -> SimResponse:
    panel_area = float(req.panel_width_m) * float(req.panel_height_m)

    orientations = _build_orientations(
        req.array_type, req.azimuth_deg, req.tilt_deg, req.total_panels
    )
    if not orientations:
        raise HTTPException(status_code=400, detail="No panel orientations defined.")

    azimuths = [o["azimuth_deg"] for o in orientations]
    panels_per_azimuth = [o["panel_count"] for o in orientations]
    tilt_deg = orientations[0]["tilt_deg"]

    result = simulate_annual_output(
        latitude=float(req.lat),
        longitude=float(req.lon),
        surface_tilt_deg=tilt_deg,
        surface_azimuths_deg=azimuths,
        panels_per_azimuth=panels_per_azimuth,
        panel_area_m2=panel_area,
        eff_stc=float(req.eff_stc),
        noct=float(req.noct),
        temp_coeff=float(req.temp_coeff),
        cooling_offset=float(req.cooling_offset),
    )
    return SimResponse(**result)



@app.post("/simulate_shaded", response_model=SimShadedResponse)
def simulate_shaded(req: SimShadedRequest) -> SimShadedResponse:
    panel_area = float(req.panel_width_m) * float(req.panel_height_m)

    orientations = _build_orientations(
        req.array_type, req.azimuth_deg, req.tilt_deg, req.total_panels
    )
    if not orientations:
        raise HTTPException(status_code=400, detail="No panel orientations defined.")
    azimuths = [o["azimuth_deg"] for o in orientations]
    panels_per_azimuth = [o["panel_count"] for o in orientations]
    tilt_deg = orientations[0]["tilt_deg"]

    year = int(req.year) if req.year else (dt.date.today().year - 1)

    pv = simulate_annual_output(
        latitude=float(req.lat),
        longitude=float(req.lon),
        surface_tilt_deg=tilt_deg,
        surface_azimuths_deg=azimuths,
        panels_per_azimuth=panels_per_azimuth,
        panel_area_m2=panel_area,
        eff_stc=float(req.eff_stc),
        noct=float(req.noct),
        temp_coeff=float(req.temp_coeff),
        cooling_offset=float(req.cooling_offset),
        year=year,
    )

    water = water_metrics_from_shading_samples(
        latitude=float(req.lat),
        longitude=float(req.lon),
        year=year,
        samples=[s.model_dump() for s in req.shading_samples],
        svf=float(req.svf) if req.svf is not None else 1.0,
    )

    merged = {**pv, **water}
    return SimShadedResponse(**merged)


@app.post("/simulate_snapshot_day", response_model=SnapshotResponse)
def simulate_snapshot_day(req: SimRequest) -> SnapshotResponse:
    orientations = _build_orientations(
        req.array_type, req.azimuth_deg, req.tilt_deg, req.total_panels
    )
    if not orientations:
        raise HTTPException(status_code=400, detail="No panel orientations defined.")

    year = dt.date.today().year - 1
    day_totals: Dict[dt.date, float] = {}
    orientation_day_stats: List[Dict[dt.date, Dict[str, float]]] = []

    for orient in orientations:
        ts = get_tilted_timeseries(
            latitude=float(req.lat),
            longitude=float(req.lon),
            tilt_deg=orient["tilt_deg"],
            azimuth_deg=orient["azimuth_api_deg"],
            year=year,
        )
        orient["timeseries"] = ts
        stats: Dict[dt.date, Dict[str, float]] = {}
        orientation_day_stats.append(stats)

        for t, g in zip(ts.times, ts.poa_w_m2):
            date = t.date()
            info = stats.setdefault(date, {"sum": 0.0, "count": 0.0})
            info["sum"] += float(g)
            info["count"] += 1.0
            day_totals[date] = day_totals.get(date, 0.0) + float(g) * orient["panel_count"]

    if not day_totals:
        raise HTTPException(status_code=400, detail="No irradiance samples available.")

    best_day = max(day_totals.items(), key=lambda kv: kv[1])[0]

    hour_totals: Dict[dt.datetime, float] = {}
    for orient in orientations:
        ts = orient["timeseries"]
        for t, g in zip(ts.times, ts.poa_w_m2):
            if t.date() != best_day:
                continue
            hour_totals[t] = hour_totals.get(t, 0.0) + float(g) * orient["panel_count"]

    if not hour_totals:
        raise HTTPException(status_code=400, detail="No hourly data for best day.")

    best_hour_dt = max(hour_totals.items(), key=lambda kv: kv[1])[0]
    zen_rad, az_rad = solar_position_approx_utc(
        t_utc=best_hour_dt,
        latitude=float(req.lat),
        longitude=float(req.lon),
    )
    el_deg = 90.0 - math.degrees(zen_rad)
    az_deg = (math.degrees(az_rad) + 360.0) % 360.0

    orientation_snapshots: List[OrientationSnapshot] = []
    for orient, stats in zip(orientations, orientation_day_stats):
        ts = orient.pop("timeseries")
        day_info = stats.get(best_day, {"sum": 0.0, "count": 0.0})
        count = day_info.get("count", 0.0)
        daily_mean = (day_info["sum"] / count) if count else 0.0
        hourly_values = [
            float(g) for t, g in zip(ts.times, ts.poa_w_m2) if t.date() == best_day
        ]

        orientation_snapshots.append(
            OrientationSnapshot(
                orientation_key=orient["orientation_key"],
                azimuth_deg=float(orient["azimuth_deg"]),
                tilt_deg=float(orient["tilt_deg"]),
                panel_count=int(orient["panel_count"]),
                daily_mean_poa_w_m2=float(daily_mean),
                daily_sum_poa_w_m2=float(day_info["sum"]),
                hourly_poa_w_m2=hourly_values,
            )
        )

    return SnapshotResponse(
        snapshot_day=best_day.isoformat(),
        snapshot_hour_utc=best_hour_dt.isoformat(),
        snapshot_sun_az_deg=float(az_deg),
        snapshot_sun_el_deg=float(el_deg),
        orientations=orientation_snapshots,
    )
# ======================= Geocoding (offline -> nominatim -> open-meteo) =======================

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

LOCAL_LOCATIONS: Dict[str, GeocodeResult] = {
    "berlin germany": GeocodeResult(display_name="Berlin, Germany", lat=52.5200, lon=13.4050),
    "turlock ca usa": GeocodeResult(display_name="Turlock, CA, USA", lat=37.4947, lon=-120.8466),
    "malta": GeocodeResult(display_name="Malta", lat=35.9375, lon=14.3754),
}


def _norm_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"[,/]+", " ", q)
    q = re.sub(r"[^a-z0-9\s\-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _offline_matches(q_norm: str, limit: int = 5) -> List[GeocodeResult]:
    if not q_norm:
        return []

    if q_norm in LOCAL_LOCATIONS:
        return [LOCAL_LOCATIONS[q_norm]]

    hits: List[GeocodeResult] = []
    for key, loc in LOCAL_LOCATIONS.items():
        if q_norm in key or key in q_norm:
            hits.append(loc)

    return hits[:limit]


def _nominatim_query(q: str, limit: int = 5) -> List[GeocodeResult]:
    params = {
        "q": q,
        "format": "json",
        "addressdetails": 1,
        "limit": limit,
    }
    headers = {
        "User-Agent": "solar-waves-sim/0.5 (contact: you@example.com)",
    }

    last_exc: Exception | None = None
    for _attempt in (1, 2):
        try:
            resp = requests.get(
                NOMINATIM_URL,
                params=params,
                headers=headers,
                timeout=(3, 12),
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail=f"Nominatim HTTP {resp.status_code}: {resp.text[:200]}",
                )
            data = resp.json()
            out: List[GeocodeResult] = []
            for item in data:
                try:
                    lat = float(item["lat"])
                    lon = float(item["lon"])
                    display = item.get("display_name") or q
                except (KeyError, ValueError):
                    continue
                out.append(GeocodeResult(display_name=display, lat=lat, lon=lon))
            return out
        except Exception as e:
            last_exc = e
            time.sleep(0.2)

    if isinstance(last_exc, HTTPException):
        raise last_exc
    raise HTTPException(
        status_code=503,
        detail=f"Nominatim unreachable: {type(last_exc).__name__}: {last_exc}",
    )


def _open_meteo_query(q: str, limit: int = 5) -> List[GeocodeResult]:
    params = {
        "name": q,
        "count": limit,
        "language": "en",
        "format": "json",
    }

    last_exc: Exception | None = None
    for _attempt in (1, 2):
        try:
            resp = requests.get(
                OPEN_METEO_GEOCODE_URL,
                params=params,
                timeout=(3, 12),
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=503,
                    detail=f"Open-Meteo HTTP {resp.status_code}: {resp.text[:200]}",
                )
            data = resp.json()
            results = []
            for item in (data.get("results") or []):
                try:
                    lat = float(item["latitude"])
                    lon = float(item["longitude"])
                    name = item.get("name") or q
                    admin1 = item.get("admin1")
                    country = item.get("country")
                    parts = [name]
                    if admin1:
                        parts.append(admin1)
                    if country:
                        parts.append(country)
                    display = ", ".join(parts)
                except (KeyError, ValueError, TypeError):
                    continue
                results.append(GeocodeResult(display_name=display, lat=lat, lon=lon))
            return results
        except Exception as e:
            last_exc = e
            time.sleep(0.2)

    if isinstance(last_exc, HTTPException):
        raise last_exc
    raise HTTPException(
        status_code=503,
        detail=f"Geocoding failed (Nominatim + Open-Meteo). Last error: {e2.detail}",
    )


def _geocode_query_with_provider(q: str, limit: int = 5) -> Tuple[str, List[GeocodeResult]]:
    q_norm = _norm_query(q)
    if not q_norm:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    offline = _offline_matches(q_norm, limit=limit)
    if offline:
        return "offline", offline

    try:
        nom = _nominatim_query(q, limit=limit)
        if nom:
            return "nominatim", nom
    except HTTPException:
        pass

    try:
        om = _open_meteo_query(q, limit=limit)
        if om:
            return "open-meteo", om
    except HTTPException as e2:
        raise HTTPException(
            status_code=503,
            detail=f"Geocoding failed (Nominatim + Open-Meteo). Last error: {e2.detail}",
        )

    raise HTTPException(status_code=404, detail="No geocoding results found")


@app.post("/geocode", response_model=GeocodeResponse)
def geocode_post(req: GeocodeRequest) -> GeocodeResponse:
    provider, results = _geocode_query_with_provider(req.query)
    return GeocodeResponse(provider=provider, results=results)


@app.get("/geocode", response_model=GeocodeResponse)
def geocode_get(q: str = Query(..., description="Location query, e.g. 'Berlin, Germany'")) -> GeocodeResponse:
    provider, results = _geocode_query_with_provider(q)
    return GeocodeResponse(provider=provider, results=results)


@app.post("/export_shaded_run")
def export_shaded_run(req: SimShadedRequest):
    """Return a ZIP 'audit pack' with inputs, shading samples, meteo hourly, and results."""
    panel_area = float(req.panel_width_m) * float(req.panel_height_m)

    if req.array_type == "waves":
        base_az = float(req.azimuth_deg)
        azimuths = [base_az - 90.0, base_az + 90.0]
        p1 = int(req.total_panels) // 2
        p2 = int(req.total_panels) - p1
        panels_per_azimuth = [p1, p2]
        tilt_deg = 12.5
    else:
        azimuths = [float(req.azimuth_deg)]
        panels_per_azimuth = [int(req.total_panels)]
        tilt_deg = float(req.tilt_deg)

    year = int(req.year) if req.year else (dt.date.today().year - 1)

    pv = simulate_annual_output(
        latitude=float(req.lat),
        longitude=float(req.lon),
        surface_tilt_deg=tilt_deg,
        surface_azimuths_deg=azimuths,
        panels_per_azimuth=panels_per_azimuth,
        panel_area_m2=panel_area,
        eff_stc=float(req.eff_stc),
        noct=float(req.noct),
        temp_coeff=float(req.temp_coeff),
        cooling_offset=float(req.cooling_offset),
        year=year,
    )

    water = water_metrics_from_shading_samples(
        latitude=float(req.lat),
        longitude=float(req.lon),
        year=year,
        samples=[s.model_dump() for s in req.shading_samples],
        svf=float(req.svf) if req.svf is not None else 1.0,
    )

    ts = get_meteo_timeseries(latitude=float(req.lat), longitude=float(req.lon), year=year)

    inputs = req.model_dump()
    inputs["year"] = year

    results = {**pv, **water}

    # Build ZIP in-memory
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("inputs.json", json.dumps(inputs, indent=2, sort_keys=True))

        # shading samples CSV
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["time_utc", "az_deg", "el_deg", "tau", "weight_hours"])
        for s in req.shading_samples:
            w.writerow([
                s.time_utc,
                "" if s.az_deg is None else float(s.az_deg),
                "" if s.el_deg is None else float(s.el_deg),
                float(s.tau),
                float(s.weight_hours),
            ])
        z.writestr("shading_samples.csv", out.getvalue())

        # SVF + sampling scheme
        z.writestr("svf.json", json.dumps({
            "svf": float(req.svf) if req.svf is not None else 1.0,
            "svf_n_rays": int(req.svf_n_rays) if req.svf_n_rays is not None else None,
            "svf_scheme": req.svf_scheme,
        }, indent=2, sort_keys=True))

        z.writestr("sampling_scheme.json", json.dumps({
            "shading_samples": {
                "description": "Representative sun samples used for GPU beam transmittance tau.",
                "fields": ["time_utc", "az_deg", "el_deg", "tau", "weight_hours"],
                "note": "az/el are optional; if absent, the time_utc can be recomputed into sun position with the documented solar position approximation.",
            },
            "svf": {
                "description": "Hemisphere raycast used to estimate diffuse sky view factor.",
                "scheme": req.svf_scheme,
                "n_rays": int(req.svf_n_rays) if req.svf_n_rays is not None else None,
            }
        }, indent=2, sort_keys=True))

        # assumptions for audit/review
        z.writestr("assumptions.md", "\n".join([
            "# Assumptions",
            "",
            "## Water-plane shortwave model",
            "",
            "Baseline (unshaded) shortwave on water is approximated as:",
            "",
            "I0 = DNI * cos(theta_z) + DHI",
            "",
            "Shaded shortwave is approximated as:",
            "",
            "I1 = DNI * cos(theta_z) * tau + DHI * SVF",
            "",
            "- tau comes from GPU shadow-mask sampling at the sun position for each representative timestamp.",
            "- SVF (sky view factor) is a scalar in [0,1] estimated via hemisphere ray-casting in the browser (isotropic diffuse assumption).",
            "",
            "Units: DNI/DHI in W/m^2; integration uses representative-hour weights and converts to kWh/m^2.",
            "",
            "## Diffuse model",
            "",
            "Diffuse is treated as isotropic and reduced linearly by SVF (first-order approximation).",
            "",
            "## Weather data",
            "",
            "Hourly radiation and meteo data are sourced from Open-Meteo archive API for the specified year.",
            "",
        ]))

        # results JSON
        z.writestr("results.json", json.dumps(results, indent=2, sort_keys=True))

        # meteo hourly CSV
        outm = io.StringIO()
        wm = csv.writer(outm)
        wm.writerow(["time_utc", "ghi_w_m2", "dni_w_m2", "dhi_w_m2", "tair_c", "rh_pct", "wind_m_s"])
        n = len(ts.times_utc)
        for i in range(n):
            t = ts.times_utc[i]
            wm.writerow([
                t.isoformat(timespec="minutes"),
                ts.ghi_w_m2[i],
                ts.dni_w_m2[i],
                ts.dhi_w_m2[i],
                ts.tair_c[i],
                ts.rh_pct[i],
                ts.wind_m_s[i],
            ])
        z.writestr("meteo_hourly.csv", outm.getvalue())

    mem.seek(0)

    fname = f"audit_pack_{year}_{float(req.lat):.3f}_{float(req.lon):.3f}.zip"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(mem, media_type="application/zip", headers=headers)
