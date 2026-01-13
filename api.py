from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests


app = FastAPI(title="Solar Output API", version="0.5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use a real contact email/URL for Nominatim usage policy compliance
# IMPORTANT: Replace contact@example.com with your email or domain.
HEADERS = {"User-Agent": "SolarWaves/1.0 (contact@example.com)"}

OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_BASE = "https://nominatim.openstreetmap.org"

# --- Simulation core ---
# sim_core.py is expected to be present in the same service.
try:
    from sim_core import simulate_annual_output, water_metrics_from_shading_samples  # type: ignore
except Exception as e:  # pragma: no cover
    simulate_annual_output = None  # type: ignore
    water_metrics_from_shading_samples = None  # type: ignore
    _SIM_IMPORT_ERROR = str(e)


class SimRequest(BaseModel):
    # Location
    lat: float
    lon: float

    # Array definition
    array_type: str = Field(default="waves")  # "waves" or "roof"
    azimuth_deg: float = Field(default=180.0)
    tilt_deg: float = Field(default=12.5)
    total_panels: int = Field(default=10, ge=1)

    # Panel geometry
    panel_width_m: float = Field(default=1.0, gt=0)
    panel_height_m: float = Field(default=1.6, gt=0)

    # Electrical / thermal parameters (optional; sim_core has defaults)
    eff_stc: Optional[float] = Field(default=None, ge=0, le=0.35)
    noct: Optional[float] = Field(default=None)
    temp_coeff: Optional[float] = Field(default=None)  # e.g. -0.0035
    cooling_offset: Optional[float] = Field(default=None)

    # Water surface (only relevant for water mode)
    water_width_m: Optional[float] = Field(default=None)


class ShadingSample(BaseModel):
    time_utc: str
    tau: float = Field(ge=0.0, le=1.0)
    weight_hours: float = Field(gt=0.0)


class SimShadedRequest(SimRequest):
    year: Optional[int] = None
    svf: float = Field(ge=0.0, le=1.0)
    svf_n_rays: Optional[int] = None
    svf_scheme: Optional[str] = None
    shading_samples: List[ShadingSample] = Field(default_factory=list)


def _open_meteo_geocode(q: str, count: int = 5) -> Dict[str, Any]:
    params = {"name": q, "count": count, "language": "en", "format": "json"}
    r = requests.get(OPEN_METEO_GEOCODE, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("results", [])[:count]:
        results.append(
            {
                "display_name": ", ".join(
                    [x for x in [item.get("name"), item.get("admin1"), item.get("country")] if x]
                ),
                "lat": float(item["latitude"]),
                "lon": float(item["longitude"]),
            }
        )
    return {"provider": "open-meteo", "results": results}


def _nominatim_reverse(lat: float, lon: float) -> Dict[str, Any]:
    url = f"{NOMINATIM_BASE}/reverse"
    params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 10, "addressdetails": 1}
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    return {
        "provider": "nominatim",
        "display_name": data.get("display_name", f"Lat {lat}, Lon {lon}"),
        "lat": float(data.get("lat", lat)),
        "lon": float(data.get("lon", lon)),
        "address": data.get("address", {}),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/geocode")
def geocode(q: str = Query(..., min_length=1), count: int = Query(5, ge=1, le=10)) -> Dict[str, Any]:
    """Forward geocode: place name -> candidate coordinates."""
    try:
        return _open_meteo_geocode(q, count=count)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"geocode_failed: {e}")


@app.get("/reverse_geocode")
def reverse_geocode(lat: float = Query(...), lon: float = Query(...)) -> Dict[str, Any]:
    """Reverse geocode: coordinates -> display name."""
    try:
        return _nominatim_reverse(lat, lon)
    except Exception as e:
        # Fail soft: still return the lat/lon so the frontend can proceed.
        return {
            "provider": "fallback",
            "display_name": f"Lat {lat}, Lon {lon}",
            "lat": float(lat),
            "lon": float(lon),
            "error": str(e),
        }


def _ensure_sim_core() -> None:
    if simulate_annual_output is None or water_metrics_from_shading_samples is None:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation core not available. Import error: {_SIM_IMPORT_ERROR}",
        )


def _compute_panel_area_m2(req: SimRequest) -> float:
    return float(req.panel_width_m) * float(req.panel_height_m)


def _compute_orientation_buckets(req: SimRequest) -> Dict[str, Any]:
    """Map frontend array_type to (surface_azimuths_deg, panels_per_azimuth).

    - Roof: single orientation at req.azimuth_deg
    - Waves: alternating +/- 90° about req.azimuth_deg (simplified, representative)
    """
    at = (req.array_type or "waves").strip().lower()
    az = float(req.azimuth_deg)

    if at in ("roof", "solar_roof", "canopy", "solar_canopy"):
        return {"surface_azimuths_deg": [az], "panels_per_azimuth": [int(req.total_panels)]}

    # Default: "waves"
    # Split evenly into two opposing faces. If odd, bias to the first bucket.
    n = int(req.total_panels)
    n1 = (n + 1) // 2
    n2 = n - n1
    return {"surface_azimuths_deg": [az - 90.0, az + 90.0], "panels_per_azimuth": [n1, n2]}


@app.post("/simulate")
def simulate(req: SimRequest) -> Dict[str, Any]:
    """Unshaded annual simulation."""
    _ensure_sim_core()

    buckets = _compute_orientation_buckets(req)

    out = simulate_annual_output(
        latitude=float(req.lat),
        longitude=float(req.lon),
        surface_tilt_deg=float(req.tilt_deg),
        surface_azimuths_deg=buckets["surface_azimuths_deg"],
        panels_per_azimuth=buckets["panels_per_azimuth"],
        panel_area_m2=_compute_panel_area_m2(req),
        eff_stc=float(req.eff_stc) if req.eff_stc is not None else 0.20,
        noct=float(req.noct) if req.noct is not None else 45.0,
        temp_coeff=float(req.temp_coeff) if req.temp_coeff is not None else -0.0035,
        cooling_offset=float(req.cooling_offset) if req.cooling_offset is not None else 0.0,
        year=None,
    )

    # The frontend expects these keys.
    return {
        "annual_energy_kwh": out.get("annual_energy_kwh"),
        "mean_poa_w_m2": out.get("mean_poa_w_m2"),
        "details": out,
    }



@app.post("/simulate_snapshot_day")
def simulate_snapshot_day(req: SimRequest) -> Dict[str, Any]:
    """Lightweight snapshot endpoint used by the frontend map picker/UI.

    The current frontend uses this to color panels for a representative day.
    This implementation returns a conservative, deterministic payload that matches
    the keys the frontend expects, without requiring additional solar-position
    dependencies.

    If you later want true day-specific outputs, this is the place to add them.
    """
    _ensure_sim_core()

    buckets = _compute_orientation_buckets(req)

    # Run the same annual model and reuse mean POA as a proxy for daily mean POA.
    # Keep arguments aligned with SimRequest fields to avoid runtime AttributeErrors.
    # simulate_annual_output in your sim_core takes these parameters.
    out = simulate_annual_output(
        latitude=float(req.lat),
        longitude=float(req.lon),
        surface_tilt_deg=float(req.tilt_deg),
        surface_azimuths_deg=buckets["surface_azimuths_deg"],
        panels_per_azimuth=buckets["panels_per_azimuth"],
        panel_area_m2=_compute_panel_area_m2(req),
        eff_stc=float(req.eff_stc) if req.eff_stc is not None else 0.20,
        noct=float(req.noct) if req.noct is not None else 45.0,
        temp_coeff=float(req.temp_coeff) if req.temp_coeff is not None else -0.0035,
        cooling_offset=float(req.cooling_offset) if req.cooling_offset is not None else 0.0,
        year=None,
    )

    mean_poa = out.get("mean_poa_w_m2")
    tilt = float(req.tilt_deg)

    def _key(az_deg: float, tilt_deg: float) -> str:
        az = az_deg % 360.0
        az_r = round(az, 1)
        t_r = round(float(tilt_deg), 1)
        return f"{az_r:.1f}|{t_r:.1f}"

    orientations = []
    for az_deg, n in zip(buckets["surface_azimuths_deg"], buckets["panels_per_azimuth"]):
        orientations.append(
            {
                "orientation_key": _key(float(az_deg), tilt),
                "panels": int(n),
                "daily_mean_poa_w_m2": mean_poa,
            }
        )

    # Provide deterministic "snapshot sun" angles so the frontend can orient shadows.
    return {
        "snapshot_sun_az_deg": float(req.azimuth_deg) % 360.0,
        "snapshot_sun_el_deg": 45.0,
        "orientations": orientations,
        "details": out,
    }



@app.post("/simulate_shaded")
def simulate_shaded(req: SimShadedRequest) -> Dict[str, Any]:
    """Shaded annual simulation (water mode): applies an effective shading factor."""
    _ensure_sim_core()

    buckets = _compute_orientation_buckets(req)

    base = simulate_annual_output(
        latitude=float(req.lat),
        longitude=float(req.lon),
        surface_tilt_deg=float(req.tilt_deg),
        surface_azimuths_deg=buckets["surface_azimuths_deg"],
        panels_per_azimuth=buckets["panels_per_azimuth"],
        panel_area_m2=_compute_panel_area_m2(req),
        eff_stc=float(req.eff_stc) if req.eff_stc is not None else 0.20,
        noct=float(req.noct) if req.noct is not None else 45.0,
        temp_coeff=float(req.temp_coeff) if req.temp_coeff is not None else -0.0035,
        cooling_offset=float(req.cooling_offset) if req.cooling_offset is not None else 0.0,
        year=int(req.year) if req.year is not None else None,
    )

    annual = float(base.get("annual_energy_kwh") or 0.0)
    mean_poa = float(base.get("mean_poa_w_m2") or 0.0)

    # Compute weighted mean transmittance tau from samples.
    # If no samples, fall back to tau=1.0 (no additional shading).
    if req.shading_samples:
        wsum = 0.0
        tsum = 0.0
        for s in req.shading_samples:
            w = float(s.weight_hours)
            t = float(s.tau)
            wsum += w
            tsum += w * t
        tau_mean = (tsum / wsum) if wsum > 0 else 1.0
    else:
        tau_mean = 1.0

    # Effective factor: combine sky-view factor and beam transmittance proxy.
    # This is a simplified model; you can refine later.
    svf = float(req.svf)
    effective = max(0.0, min(1.0, svf * tau_mean))

    # Water-plane metrics expected by the frontend
    year_int = int(req.year) if req.year is not None else None
    # Default to 2024 if year not provided (last full year), keeps results deterministic.
    if year_int is None:
        year_int = 2024

    wm = water_metrics_from_shading_samples(
        latitude=float(req.lat),
        longitude=float(req.lon),
        year=year_int,
        samples=[s.model_dump() for s in (req.shading_samples or [])],
        svf=svf,
    )

    return {
        # PV energy/POA should come from the PV model (base) and should not be scaled by water SVF/tau.
        "annual_energy_kwh": annual,
        "mean_poa_w_m2": mean_poa,

        # Water metrics used for savings scaling in water mode
        "water_baseline_kwh_m2": float(wm.get("water_baseline_kwh_m2", 0.0)),
        "water_shaded_kwh_m2": float(wm.get("water_shaded_kwh_m2", 0.0)),
        "water_reduction_pct": float(wm.get("water_reduction_pct", 0.0)),

        # Debug shading fields
        "shading": {
            "svf": svf,
            "tau_mean": tau_mean,
            "effective_factor": effective,
            "n_samples": len(req.shading_samples),
            "svf_n_rays": req.svf_n_rays,
            "svf_scheme": req.svf_scheme,
        },
        "details": base,
    }
